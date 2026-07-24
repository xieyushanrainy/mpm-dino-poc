from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import dataset_records, make_split_manifest, prepare_cache, validate_manifest
from .evaluate import evaluate, resummarize
from .full_evaluate import compare_tracks, evaluate_full
from .full_train import train_full_model
from .train import train_model


def read_manifest(path):
    value=json.loads(Path(path).read_text()); validate_manifest(value); return value


def audit_dataset(root):
    root=Path(root); rows=[]
    for record in dataset_records(root):
        import numpy as np
        metadata=json.loads((root/record["metadata"]).read_text())
        with np.load(root/record["sample"]) as sample:
            x=sample["trajectory_positions_m"]; active=sample["point_active"]
            com=np.asarray([x[t,mask].mean(0) if mask.any() else [np.nan]*3 for t,mask in enumerate(active)])
            rows.append({"uid":record["uid"],"family":record["solver_route"],"frame_0_1_com_drop_mm":float((com[0,2]-com[1,2])*1000),
                         "frame_1_min_z_m":float(x[1,:,2].min()),"frame_1_below_floor_fraction":float((x[1,:,2]<0).mean()),
                         "minimum_active_fraction":float(active.mean(1).min()),"dino_valid_fraction":float(sample["dino_valid"].mean()),
                         "dt_s":float(metadata["time"]["time_step_s"]),"trajectory_semantics":metadata["simulation"]["point_trajectory_semantics"]})
    summary={}
    for family in sorted({row["family"] for row in rows}):
        group=[row for row in rows if row["family"]==family]
        summary[family]={"objects":len(group),"median_frame_0_1_com_drop_mm":float(np.median([r["frame_0_1_com_drop_mm"] for r in group])),
                         "objects_below_floor_at_frame_1":sum(r["frame_1_min_z_m"]<0 for r in group),
                         "median_minimum_active_fraction":float(np.median([r["minimum_active_fraction"] for r in group])),
                         "median_dino_valid_fraction":float(np.median([r["dino_valid_fraction"] for r in group]))}
    return {"summary":summary,"fluid_training_gate":"FAIL: regenerate before three-family claims","objects":rows}


def aggregate(paths, output):
    runs=[json.loads(Path(path).read_text()) for path in paths]; grouped=defaultdict(list)
    for run in runs:
        for key,value in run["rollout"].items():
            if key.startswith("object_weighted/") and key.endswith(("/H1","/H4","/H8","/H16")):
                grouped[(run["dino_mode"],key)].append(value["rmse_m"])
    summary={f"{mode}/{key}":{"mean_rmse_m":float(np.mean(values)),"std_rmse_m":float(np.std(values,ddof=1)) if len(values)>1 else 0.0,"seeds":len(values)} for (mode,key),values in grouped.items()}
    verdict="insufficient matched real/zero controls"
    key="object_weighted/aggregate/H8"
    if all((mode,key) in grouped for mode in ("real","zero","scene_shuffled")):
        real,zero,shuffle=(np.asarray(grouped[(mode,key)]) for mode in ("real","zero","scene_shuffled"))
        pooled=max(np.std(real,ddof=1),np.std(zero,ddof=1),np.std(shuffle,ddof=1)) if min(map(len,(real,zero,shuffle)))>1 else float("inf")
        margin=min(zero.mean()-real.mean(),shuffle.mean()-real.mean())
        verdict="DINO passes matched H8 seed-variation criterion" if margin>pooled else "DINO benefit not established beyond seed variation"
    elif all((mode,key) in grouped for mode in ("real","zero")):
        real,zero=(np.asarray(grouped[(mode,key)]) for mode in ("real","zero"))
        pooled=max(np.std(real,ddof=1),np.std(zero,ddof=1)) if min(len(real),len(zero))>1 else float("inf")
        margin=zero.mean()-real.mean()
        verdict=("Real DINO beats zero DINO beyond H8 seed variation; shuffled-control evidence is still absent"
                 if margin>pooled else "Real DINO does not beat zero DINO beyond H8 seed variation")
    payload={"runs":[str(Path(p).resolve()) for p in paths],"summary":summary,"verdict":verdict}
    Path(output).write_text(json.dumps(payload,indent=2)+"\n"); return payload


def parser():
    result=argparse.ArgumentParser(); sub=result.add_subparsers(dest="command",required=True)
    audit=sub.add_parser("audit"); audit.add_argument("--dataset",required=True); audit.add_argument("--output",required=True)
    split=sub.add_parser("split"); split.add_argument("--dataset",required=True); split.add_argument("--output",required=True); split.add_argument("--seed",type=int,default=20260722)
    prep=sub.add_parser("prepare"); prep.add_argument("--dataset",required=True); prep.add_argument("--manifest",required=True); prep.add_argument("--output",required=True)
    train=sub.add_parser("train"); train.add_argument("--cache",required=True); train.add_argument("--manifest",required=True); train.add_argument("--output",required=True); train.add_argument("--dino-mode",choices=("real","zero","scene_shuffled","point_shuffled"),required=True); train.add_argument("--seed",type=int,required=True); train.add_argument("--device",default="mps"); train.add_argument("--epochs",type=int,default=60); train.add_argument("--batch-size",type=int,default=2); train.add_argument("--hidden-dim",type=int,default=128); train.add_argument("--layers",type=int,default=3); train.add_argument("--max-batches",type=int)
    ev=sub.add_parser("evaluate"); ev.add_argument("--cache",required=True); ev.add_argument("--manifest",required=True); ev.add_argument("--output",required=True); ev.add_argument("--checkpoint"); ev.add_argument("--dino-mode",choices=("real","zero","scene_shuffled","point_shuffled"),default="zero"); ev.add_argument("--seed",type=int,default=42); ev.add_argument("--split",choices=("train","validation","test"),default="test"); ev.add_argument("--device",default="mps"); ev.add_argument("--families",nargs="+",choices=("rigid","fluid","soft_body"),default=("rigid","soft_body")); ev.add_argument("--batch-size",type=int,default=2); ev.add_argument("--max-starts",type=int); ev.add_argument("--max-horizon",type=int)
    agg=sub.add_parser("aggregate"); agg.add_argument("metrics",nargs="+"); agg.add_argument("--output",required=True)
    repair=sub.add_parser("resummarize"); repair.add_argument("metrics",nargs="+")
    full_train=sub.add_parser("train-full"); full_train.add_argument("--cache",required=True); full_train.add_argument("--manifest",required=True); full_train.add_argument("--output",required=True); full_train.add_argument("--dino-mode",choices=("real","zero","scene_shuffled"),required=True); full_train.add_argument("--seed",type=int,required=True); full_train.add_argument("--device",default="mps"); full_train.add_argument("--epochs",type=int,default=300); full_train.add_argument("--batch-size",type=int,default=1); full_train.add_argument("--accumulation-steps",type=int,default=4); full_train.add_argument("--hidden-dim",type=int,default=128); full_train.add_argument("--blocks",type=int,default=4); full_train.add_argument("--heads",type=int,default=4); full_train.add_argument("--dropout",type=float,default=0.1); full_train.add_argument("--max-batches",type=int)
    full_eval=sub.add_parser("evaluate-full"); full_eval.add_argument("--cache",required=True); full_eval.add_argument("--manifest",required=True); full_eval.add_argument("--output",required=True); full_eval.add_argument("--checkpoint"); full_eval.add_argument("--baseline",choices=("ballistic","constant_velocity")); full_eval.add_argument("--dino-mode",choices=("real","zero","scene_shuffled"),default="zero"); full_eval.add_argument("--seed",type=int,default=42); full_eval.add_argument("--split",choices=("train","validation","test"),default="test"); full_eval.add_argument("--device",default="mps"); full_eval.add_argument("--families",nargs="+",choices=("rigid","soft_body"),default=("rigid","soft_body"))
    compare=sub.add_parser("compare-tracks"); compare.add_argument("--track-b",nargs="+",required=True); compare.add_argument("--track-a",nargs="+",required=True); compare.add_argument("--baselines",nargs="+",required=True); compare.add_argument("--output",required=True)
    return result


def main():
    args=parser().parse_args()
    if args.command=="audit": payload=audit_dataset(args.dataset); Path(args.output).write_text(json.dumps(payload,indent=2)+"\n")
    elif args.command=="split": payload=make_split_manifest(args.dataset,args.seed); validate_manifest(payload); Path(args.output).write_text(json.dumps(payload,indent=2)+"\n")
    elif args.command=="prepare": prepare_cache(args.dataset,read_manifest(args.manifest),args.output)
    elif args.command=="train": train_model(args.cache,read_manifest(args.manifest),args.output,args.dino_mode,args.seed,args.device,args.epochs,args.batch_size,args.hidden_dim,args.layers,max_batches=args.max_batches)
    elif args.command=="evaluate": evaluate(args.cache,read_manifest(args.manifest),args.split,args.dino_mode,args.seed,args.output,args.checkpoint,args.device,tuple(args.families),args.batch_size,args.max_starts,args.max_horizon)
    elif args.command=="aggregate": print(json.dumps(aggregate(args.metrics,args.output),indent=2))
    elif args.command=="resummarize":
        for path in args.metrics: resummarize(path)
    elif args.command=="train-full":
        train_full_model(args.cache,read_manifest(args.manifest),args.output,args.dino_mode,args.seed,args.device,args.epochs,args.batch_size,args.accumulation_steps,args.hidden_dim,args.blocks,args.heads,args.dropout,max_batches=args.max_batches)
    elif args.command=="evaluate-full":
        evaluate_full(args.cache,read_manifest(args.manifest),args.split,args.dino_mode,args.seed,args.output,args.checkpoint,args.baseline,args.device,tuple(args.families))
    elif args.command=="compare-tracks":
        compare_tracks(args.track_b,args.track_a,args.baselines,args.output)


if __name__=="__main__": main()
