from __future__ import annotations

import torch

from mpm_dino_v2.graph import build_mutual_knn_graph, validate_neighbour_graph
from mpm_dino_v4.data import WindowDataset, donor_map, point_permutation, validate_manifest
from mpm_dino_v4.losses import compute_loss
from mpm_dino_v4.metrics import metric_values
from mpm_dino_v4.model import V4ParticleSurrogate, masked_mean


def inputs(n=16):
    torch.manual_seed(7); reference=torch.rand(n,3); graph=build_mutual_knn_graph(reference,torch.ones(n,dtype=torch.bool),4,3)
    x_prev=reference.clone(); x_curr=reference+torch.tensor([0.,0.,-.01]); mask=torch.ones(n,dtype=torch.bool)
    batch={"x_prev":x_prev[None],"x_curr":x_curr[None],"mask_prev":mask[None],"mask_curr":mask[None],
           "reference":reference[None],"dino":torch.randn(1,n,384),"dino_valid":torch.arange(n)[None]%3!=0,
           "dt":torch.tensor([1/30]),"gravity":torch.tensor([[0.,0.,-9.81]]),"floor_z":torch.tensor([0.]),
           **{key:value[None] for key,value in graph.items()}}
    return batch,graph


def test_cv_zero_initialization_and_local_zero_mean():
    batch,_=inputs(); model=V4ParticleSurrogate(hidden_dim=32,layers=2,dino_embed_dim=8); output=model(**batch)
    assert torch.allclose(output.position,2*batch["x_curr"]-batch["x_prev"])
    assert torch.allclose(masked_mean(output.residual_local,batch["mask_curr"]),torch.zeros(1,3),atol=1e-7)


def test_loss_masks_inactive_and_backpropagates():
    batch,_=inputs(); model=V4ParticleSurrogate(hidden_dim=32,layers=1,dino_embed_dim=8); output=model(**batch)
    target=output.cv_position.clone(); target[:,0]=1000; loss_batch={**batch,"target":target,"target_mask":batch["mask_curr"].clone()}; loss_batch["target_mask"][:,0]=False
    loss=compute_loss(output,loss_batch); assert float(loss.total.detach())<1e-8; loss.total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_graph_and_shuffle_controls_are_deterministic():
    batch,graph=inputs(); report=validate_neighbour_graph(batch["reference"][0],batch["mask_curr"][0],**graph); assert report.directed_edge_count>0
    assert donor_map(["a","b","c"],42)==donor_map(["a","b","c"],42)
    donors=donor_map(["a","b","c"],42); assert all(uid!=donor for uid,donor in donors.items())
    assert torch.equal(point_permutation("a",20,42),point_permutation("a",20,42))


def test_manifest_leakage_rejected():
    manifest={"splits":{"train":["x"],"validation":["x"],"test":[]},"strata":{}}
    try: validate_manifest(manifest)
    except ValueError as error: assert "leakage" in str(error)
    else: raise AssertionError("leaky manifest accepted")


def test_metric_fixture_exact_prediction():
    batch,_=inputs(); target=2*batch["x_curr"]-batch["x_prev"]
    values=metric_values(target,target,batch["mask_curr"],batch["reference"],batch["neighbour_indices"],batch["neighbour_mask"],batch["floor_z"],["rigid"])
    for key in ("rmse_m","mae_m","com_m","shape_m","edge_vector_m","edge_length_m"):
        assert torch.allclose(values[key],torch.zeros_like(values[key]),atol=1e-7)


def test_sliding_window_boundaries_and_active_intersection(tmp_path):
    n=8; reference=torch.rand(n,3); graph=build_mutual_knn_graph(reference,torch.ones(n,dtype=torch.bool),4,3)
    positions=torch.stack([reference+torch.tensor([0.,0.,-.01*t]) for t in range(4)])
    active=torch.ones(4,n,dtype=torch.bool); active[2,0]=False
    scene={"uid":"u","family":"rigid","category":"test","positions":positions,"active":active,
           "dino":torch.randn(n,384),"dino_valid":torch.ones(n,dtype=torch.bool),"dt":1/30,
           "gravity":torch.tensor([0.,0.,-9.81]),"floor_z":0.,**graph}
    torch.save(scene,tmp_path/"u.pt")
    manifest={"splits":{"train":["u"],"validation":[],"test":[]},"strata":{"u":{"family":"rigid"}}}
    dataset=WindowDataset(tmp_path,manifest,"train",("rigid",),"real",42)
    assert len(dataset)==2
    assert dataset[0]["t"]==0 and torch.allclose(dataset[0]["target"],positions[2])
    assert not bool(dataset[0]["target_mask"][0])
    assert dataset[1]["t"]==1 and torch.allclose(dataset[1]["target"],positions[3])
