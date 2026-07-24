from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, point_permutation
from .metrics import metric_values
from .train import MODEL_KEYS, load_model, move


def _append(rows, batch, pred, target, horizon, baseline):
    values = metric_values(pred, target, batch["target_mask"], batch["reference"], batch["neighbour_indices"],
                           batch["neighbour_mask"], batch["floor_z"], list(batch["family"]))
    cv = metric_values(baseline, target, batch["target_mask"], batch["reference"], batch["neighbour_indices"],
                       batch["neighbour_mask"], batch["floor_z"], list(batch["family"]))
    for index, uid in enumerate(batch["uid"]):
        row = {"uid": uid, "family": batch["family"][index], "start_t": int(batch["t"][index]), "horizon": horizon,
               "active_points": int(batch["target_mask"][index].sum().detach().cpu())}
        for key, value in values.items(): row[key] = float(value[index].detach().cpu())
        row["cv_rmse_m"] = float(cv["rmse_m"][index].detach().cpu())
        row["cv_relative_improvement"] = 1.0 - row["rmse_m"] / max(row["cv_rmse_m"], 1e-12)
        rows.append(row)


def one_step(model, dataset, device, batch_size=2):
    rows=[]
    for raw in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch=move(raw, device)
        with torch.no_grad(): output=model(**{key:batch[key] for key in MODEL_KEYS}) if model else None
        pred=output.position if output else 2*batch["x_curr"]-batch["x_prev"]
        _append(rows,batch,pred,batch["target"],1,2*batch["x_curr"]-batch["x_prev"])
    return rows


def _dino_for(dataset, uid, scene):
    donor = dataset._load(dataset.donors[uid]) if dataset.dino_mode == "scene_shuffled" else scene
    dino, valid = donor["dino"], donor["dino_valid"]
    if dataset.dino_mode == "zero": dino=torch.zeros_like(dino)
    elif dataset.dino_mode == "point_shuffled":
        order=point_permutation(uid,len(dino),dataset.seed); dino,valid=dino[order],valid[order]
    return dino,valid


def rollout(model, dataset, device, max_starts=None, max_horizon=None):
    rows=[]
    for uid in dataset.uids:
        scene=dataset._load(uid); dino,valid=_dino_for(dataset,uid,scene); positions=scene["positions"]; active=scene["active"]
        starts = range(positions.shape[0]-2) if max_starts is None else range(min(max_starts, positions.shape[0]-2))
        for start in starts:
            previous,current=positions[start].to(device),positions[start+1].to(device)
            cv_previous,cv_current=previous.clone(),current.clone()
            state_mask=(active[start]&active[start+1]).to(device)
            for target_index in range(start+2,positions.shape[0]):
                horizon=target_index-(start+1); target=positions[target_index].to(device); target_mask=(state_mask.cpu()&active[target_index]).to(device)
                if max_horizon is not None and horizon > max_horizon: break
                if model:
                    inputs={"x_prev":previous[None],"x_curr":current[None],"mask_prev":state_mask[None],"mask_curr":state_mask[None],
                            "reference":positions[0][None].to(device),"dino":dino[None].to(device),"dino_valid":valid[None].to(device),
                            "dt":torch.tensor([scene["dt"]],device=device),"gravity":scene["gravity"][None].to(device),
                            "floor_z":torch.tensor([scene["floor_z"]],device=device),
                            **{key:scene[key][None].to(device) for key in ("neighbour_indices","neighbour_mask","rest_edge_vectors","rest_edge_lengths")}}
                    with torch.no_grad(): prediction=model(**inputs).position[0]
                else: prediction=2*current-previous
                cv_prediction=2*cv_current-cv_previous
                batch={"uid":[uid],"family":[scene["family"]],"t":torch.tensor([start]),"target_mask":target_mask[None],
                       "reference":positions[0][None].to(device),"neighbour_indices":scene["neighbour_indices"][None].to(device),
                       "neighbour_mask":scene["neighbour_mask"][None].to(device),"floor_z":torch.tensor([scene["floor_z"]],device=device)}
                _append(rows,batch,prediction[None],target[None],horizon,cv_prediction[None])
                previous,current=current,prediction; cv_previous,cv_current=cv_current,cv_prediction
    return rows


def summarize(rows):
    def finite_mean(values):
        finite=[value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")
    result={}
    for weighting in ("point_weighted","window_weighted","object_weighted"):
        for family in sorted({row["family"] for row in rows}|{"aggregate"}):
            selected=rows if family=="aggregate" else [row for row in rows if row["family"]==family]
            for horizon in sorted({row["horizon"] for row in selected}):
                group=[row for row in selected if row["horizon"]==horizon]; metrics={}
                for key in ("rmse_m","mae_m","com_m","shape_m","edge_vector_m","edge_length_m","floor_penetration_rate","floor_penetration_depth_m","active_coverage","rigidity_residual_m","cv_rmse_m","cv_relative_improvement"):
                    if weighting=="window_weighted": values=[row[key] for row in group]
                    elif weighting=="point_weighted":
                        values=[row[key] for row in group]; weights=np.asarray([row["active_points"] for row in group],dtype=float)
                        finite=np.asarray([np.isfinite(value) for value in values]); values_array=np.asarray(values,dtype=float)
                        if finite.any():
                            if key in {"rmse_m","cv_rmse_m"}: metrics[key]=float(np.sqrt(np.average(values_array[finite]**2,weights=weights[finite])))
                            else: metrics[key]=float(np.average(values_array[finite],weights=weights[finite]))
                        else: metrics[key]=float("nan")
                        continue
                    else:
                        by_uid=defaultdict(list)
                        for row in group: by_uid[row["uid"]].append(row[key])
                        values=[finite_mean(value) for value in by_uid.values()]
                    metrics[key]=finite_mean(values)
                metrics["cv_relative_improvement"] = 1.0 - metrics["rmse_m"] / max(metrics["cv_rmse_m"], 1e-12)
                result[f"{weighting}/{family}/H{horizon}"]={"count":len(group),**metrics}
    return result


def resummarize(path):
    path=Path(path); payload=json.loads(path.read_text())
    payload["one_step"]=summarize(payload["object_rows"]); payload["rollout"]=summarize(payload["rollout_rows"])
    path.write_text(json.dumps(payload,indent=2,allow_nan=True)+"\n")
    return payload


def evaluate(cache,manifest,split,dino_mode,seed,output,checkpoint=None,device="mps",families=("rigid","soft_body"),batch_size=2,max_starts=None,max_horizon=None):
    dataset=WindowDataset(cache,manifest,split,families,dino_mode,seed)
    model=None
    if checkpoint:
        model,config=load_model(checkpoint,device)
        if config["dino_mode"] != dino_mode or int(config["seed"]) != seed:
            raise ValueError("evaluation DINO mode and seed must match the checkpoint configuration")
        model.eval()
    rows_one=one_step(model,dataset,device,batch_size); rows_roll=rollout(model,dataset,device,max_starts,max_horizon)
    payload={"checkpoint":str(checkpoint) if checkpoint else None,"dino_mode":dino_mode,"seed":seed,"split":split,
             "families":list(families),"one_step":summarize(rows_one),"rollout":summarize(rows_roll),"object_rows":rows_one,"rollout_rows":rows_roll}
    Path(output).write_text(json.dumps(payload,indent=2,allow_nan=True)+"\n"); return payload
