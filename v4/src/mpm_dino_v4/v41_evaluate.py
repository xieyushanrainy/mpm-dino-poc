from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import metric_values
from .v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset
from .v41_model import V41TrajectorySurrogate
from .v41_train import move


def load_model(path, device):
    state=torch.load(path,map_location="cpu",weights_only=False); c=state["config"]
    model=V41TrajectorySurrogate(c["mechanism"],hidden_dim=c["hidden_dim"],blocks=c["blocks"],
                                 heads=c["heads"],dropout=c["dropout"]).to(device)
    model.load_state_dict(state["model"]); return model.eval(),c


def evaluate_v41(root, manifest, checkpoint, output, split="test", device="mps"):
    model,config=load_model(checkpoint,device)
    ds=V41TrajectoryDataset(root,manifest,split,config["dino_mode"],config["seed"])
    rows=[]
    with torch.no_grad():
        for raw in DataLoader(ds,batch_size=1,shuffle=False):
            batch=move(raw,torch.device(device))
            pred=model(**{k:batch[k] for k in MODEL_INPUT_KEYS}).position
            for i in range(pred.shape[1]):
                values=metric_values(pred[:,i],batch["target"][:,i],batch["target_mask"][:,i],
                    batch["reference"],batch["neighbour_indices"],batch["neighbour_mask"],
                    batch["floor_z"],list(batch["family"]))
                row={"uid":batch["uid"][0],"episode_id":batch["episode_id"][0],
                     "family":batch["family"][0],"panel":batch["panel"][0],
                     "velocity_regime":batch["velocity_regime"][0],"initial_velocity":batch["initial_velocity"][0].cpu().tolist(),
                     "horizon":i+1,"active_points":int(batch["target_mask"][0,i].sum().cpu())}
                row.update({k:float(v[0].cpu()) for k,v in values.items()}); rows.append(row)
    summary={}
    for panel in ("Z","V"):
        selected=[r for r in rows if r["panel"]==panel]
        for family in sorted({r["family"] for r in selected}|{"aggregate"}):
            group=selected if family=="aggregate" else [r for r in selected if r["family"]==family]
            for h in (1,8,16,30,40,59):
                hrows=[r for r in group if r["horizon"]==h]
                if not hrows: continue
                by_uid=defaultdict(list)
                for r in hrows: by_uid[r["uid"]].append(r)
                metrics={}
                for key in ("rmse_m","mae_m","com_m","shape_m","edge_vector_m","edge_length_m",
                            "floor_penetration_rate","floor_penetration_depth_m","active_coverage","rigidity_residual_m"):
                    uid_values=[np.nanmean([r[key] for r in values]) for values in by_uid.values()]
                    metrics[key]=float(np.nanmean(uid_values))
                summary[f"panel_{panel}/{family}/H{h}"]={"uids":len(by_uid),"episodes":len(hrows),**metrics}
        curve=[]
        for h in range(1,60):
            vals=[r["rmse_m"] for r in selected if r["horizon"]==h]
            curve.append({"horizon":h,"rmse_m":float(np.mean(vals)) if vals else float("nan")})
        summary[f"panel_{panel}/per_frame_curve"]=curve
    payload={"checkpoint":str(Path(checkpoint).resolve()),"config":config,"split":split,
             "panels_separate":True,"summary":summary,"object_rows":rows}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(payload,indent=2,allow_nan=True)+"\n")
    return payload


def aggregate_v41(paths, output):
    runs=[json.loads(Path(p).read_text()) for p in paths]
    result={"panels_separate":True,"mechanisms":{}}
    for mechanism in sorted({r["config"]["mechanism"] for r in runs}):
        result["mechanisms"][mechanism]={}
        subset=[r for r in runs if r["config"]["mechanism"]==mechanism]
        for panel in ("Z","V"):
            for h in (1,8,16,30,40,59):
                key=f"panel_{panel}/aggregate/H{h}"
                modes={}
                for mode in ("real","zero","scene_shuffled","point_shuffled"):
                    vals=[r["summary"][key]["rmse_m"] for r in subset if r["config"]["dino_mode"]==mode and key in r["summary"]]
                    if vals: modes[mode]={"mean_rmse_m":float(np.mean(vals)),"seeds":len(vals),"values":vals}
                if modes: result["mechanisms"][mechanism][f"panel_{panel}/H{h}"]=modes
        real=[r for r in subset if r["config"]["dino_mode"]=="real"]
        zero=[r for r in subset if r["config"]["dino_mode"]=="zero"]
        verdict={"promoted":False,"reason":"incomplete three-seed matched matrix"}
        if len(real)==len(zero)==3:
            for h in (30,40):
                rk="panel_Z/aggregate/H"+str(h)
                rv={r["config"]["seed"]:r["summary"][rk]["rmse_m"] for r in real}
                zv={r["config"]["seed"]:r["summary"][rk]["rmse_m"] for r in zero}
                wins=sum(rv[s]<zv[s] for s in rv)
                h1r=np.mean([r["summary"]["panel_Z/aggregate/H1"]["rmse_m"] for r in real])
                h1z=np.mean([r["summary"]["panel_Z/aggregate/H1"]["rmse_m"] for r in zero])
                if np.mean(list(rv.values()))<np.mean(list(zv.values())) and wins>=2 and h1r<=1.1*h1z:
                    verdict={"promoted":True,"horizon":h,"paired_seed_wins":wins}; break
            if not verdict["promoted"]: verdict={"promoted":False,"reason":"promotion rule not met"}
        result["mechanisms"][mechanism]["promotion"]=verdict
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=True)+"\n")
    return result
