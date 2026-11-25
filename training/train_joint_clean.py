"""Joint training with a separate predictor for each annotation vocabulary."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data import SeparateHelmetDataset, collate_detection
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold
from evaluate_paired_70_73 import reserve
from models import build_detector, forbid_external_weights
from train_clean_heads import dataset


@torch.inference_mode()
def evaluate(model, samples):
    model.eval()
    predictions, targets = [], []
    for images, labels in samples:
        predictions.extend({k:v.cpu() for k,v in p.items()} for p in model([x.cuda() for x in images]))
        targets.extend(labels)
    return dict(metrics=detection_ap50(predictions,targets,(1,2,3,4)), counts=counts_at_threshold(predictions,targets))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--multiscale-six-sources',action='store_true')
    parser.add_argument('--head-context',action='store_true')
    parser.add_argument('--repeat',action='store_true')
    parser.add_argument('--occupants',action='store_true')
    parser.add_argument('--lowrate-parent',choices=('joint-occupants-04','joint-multiscale-02'))
    parser.add_argument('--low-detail',action='store_true')
    args=parser.parse_args()
    if args.lowrate_parent:
        args.occupants=args.lowrate_parent=='joint-occupants-04'
        args.multiscale_six_sources=True
    context=args.head_context
    large=args.multiscale_six_sources or context or args.occupants or args.low_detail
    seed=20261124
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    forbid_external_weights()
    run=Path("runs/joint-multiscale-02" if large else "runs/joint-clean640-01")
    config_parent="runs/joint-clean640-01/best.pt" if large else "runs/detector-clean-full-01/last.pt"
    if context:
        run=Path("runs/joint-head-context-03-repeat" if args.repeat else "runs/joint-head-context-03")
        config_parent="runs/joint-multiscale-02/best.pt"
    if args.occupants:
        run=Path("runs/joint-occupants-04")
        config_parent="runs/joint-multiscale-02/best.pt"
    if args.lowrate_parent:
        config_parent=f"runs/{args.lowrate_parent}/best.pt"
        run=Path("runs/joint-lowrate-05")
    if args.low_detail:
        config_parent="runs/joint-multiscale-02/best.pt"
        run=Path("runs/joint-low-detail-06")
    run.mkdir(exist_ok=False)
    root="data/prepared/helmet-separate-v2-expanded-09"
    full=SeparateHelmetDataset(root,"train",photometric=True)
    val=SeparateHelmetDataset(root,"val",augment=False)
    edge=dataset("train")
    full.low_detail=edge.low_detail=args.low_detail
    full.image_size=edge.image_size=768 if large else 640
    assert not {r["group_id"] for r in val.rows} & ({r["groupId"] for r in edge.rows}|{r["group_id"] for r in full.rows})
    from head_context import context_batch
    loaders=[DataLoader(d,batch_size=b,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True,collate_fn=context_batch if context and d is edge else collate_detection)
             for d,b in ((edge,6),(full,3))]
    if args.occupants:
        cropped=DataLoader(edge,batch_size=6,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=True,collate_fn=context_batch)
    validation=DataLoader(val,batch_size=2,collate_fn=collate_detection)
    model=build_detector(5,8,4).cuda()
    src=torch.load(config_parent,weights_only=True,map_location="cpu")
    model.load_state_dict(src["model"])
    primary=model.roi_heads.box_predictor
    auxiliary=FastRCNNPredictor(primary.cls_score.in_features,4).cuda()
    prior=torch.load("runs/edge-clean-20261124/transfer.pt",weights_only=True,map_location="cpu")
    auxiliary.load_state_dict({k.split("roi_heads.box_predictor.")[1]:v for k,v in prior["model"].items() if k.startswith("roi_heads.box_predictor.")})
    if large:
        from train_joint_consistency import Photos, photo_batch, consistent_loss
        data=Photos(run)
        assert len(data)==7222
        photoloader=DataLoader(data,batch_size=8,num_workers=2,shuffle=True,persistent_workers=True,pin_memory=True,collate_fn=photo_batch)
        photos=iter(photoloader)
        teacher=copy.deepcopy(model).eval()
        with torch.no_grad():
            for old,new in ((0,0),(4,2),(3,3)):
                for name,width in (("cls_score",1),("bbox_pred",4)):
                    dst=getattr(auxiliary,name)
                    ref=getattr(primary,name)
                    for param in ("weight","bias"):
                        getattr(dst,param)[new*width:(new+1)*width].copy_(getattr(ref,param)[old*width:(old+1)*width])
    average=copy.deepcopy(model)
    optimizer=torch.optim.AdamW([{"params":model.parameters(),"lr":.000018 if args.lowrate_parent else .000045},{"params":auxiliary.parameters(),"lr":.000075 if args.lowrate_parent else .00015}],weight_decay=.0001)
    epochs=30 if context else (24 if large else 12)
    if args.occupants:
        epochs=32
    if args.lowrate_parent:
        epochs=36
    if args.low_detail:
        epochs=8
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.000006)
    scaler=torch.amp.GradScaler("cuda")
    cfg=dict(src["config"],dataset=root,run=run.name,image_size=640,deterministic_gpu=False,
             parent_sha256=hashlib.sha256(Path(config_parent).read_bytes()).hexdigest(),initialization="joint_multiscale_transfer" if large else "joint_project_transfer",train_examples=len(full),auxiliary_examples=len(edge),external_pretrained=False)
    if context:
        cfg["initialization"]="joint_context_transfer"
    if args.occupants:
        cfg["initialization"]="joint_context_transfer"
    if args.lowrate_parent:
        cfg.update(initialization="joint_lowrate_transfer",parent_run=args.lowrate_parent)
    if args.low_detail:
        cfg.update(initialization='joint_context_transfer',low_detail=True)
    config=dict(low_detail=args.low_detail,learning_rates=[g["lr"] for g in optimizer.param_groups],occupant_replay=args.occupants,head_context=context,parent=config_parent,consistency_examples=7222 if large else 0,training_sizes=[576,640,768] if large else [640],epochs=epochs,primary=len(full),auxiliary=len(edge),selection="highest validation four-class mAP50")
    (run/"protocol.json").write_text(json.dumps(config,indent=2))
    best=-1
    history=[]
    reviewed=iter(loaders[1])
    updates=0
    for epoch in range(1,epochs+1):
        model.train()
        for bn in model.modules():
            if isinstance(bn,torch.nn.BatchNorm2d): bn.eval()
        loss_sum=0.
        aux_loader=cropped if args.occupants and epoch%2==0 else loaders[0]
        batch_count=0
        for step,aux_batch in enumerate(aux_loader):
            batches=[(auxiliary,aux_batch)]
            if (step%3!=2 if args.occupants else step%2==0):
                try: batch=next(reviewed)
                except StopIteration:
                    reviewed=iter(loaders[1])
                    batch=next(reviewed)
                batches.append((primary,batch))
            for predictor,(images,targets) in batches:
                model.roi_heads.box_predictor=predictor
                if large:
                    size=random.choice((576,640,768))
                    model.transform.min_size=(round(.6*size),)
                    model.transform.max_size=size
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    parts=model([x.cuda(non_blocking=True) for x in images],[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets])
                    loss=sum(parts.values())
                    if args.occupants and predictor is primary:
                        loss=loss+.25*parts['loss_box_reg']
                    if large and predictor is primary and step%4==0:
                        try: unlabeled=next(photos)
                        except StopIteration:
                            photos=iter(photoloader)
                            unlabeled=next(photos)
                        model.transform.min_size=(384,)
                        model.transform.max_size=640
                        loss=loss+.1*consistent_loss(model,teacher,[x.cuda(non_blocking=True) for x in unlabeled])
                if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(optimizer)
                scaler.update()
                batch_count+=1
                updates+=1
                loss_sum+=loss.item()
                model.roi_heads.box_predictor=primary
                with torch.no_grad():
                    for a,p in zip(average.parameters(),model.parameters()):
                        a.lerp_(p,.003)
        scheduler.step()
        result=evaluate(average,validation)
        score=result["metrics"]["map50"]
        history.append(dict(epoch=epoch,updates=updates,loss=loss_sum/batch_count,**result))
        (run/"history.json").write_text(json.dumps(history,indent=2))
        if score>best:
            best=score
            torch.save(dict(model=average.state_dict(),epoch=epoch,config=cfg,validation=result["metrics"],optimizer_updates=updates),run/"best.pt")
        print(epoch,updates,score,flush=True)
    ck=torch.load(run/"best.pt",weights_only=True)
    average.load_state_dict(ck["model"])
    held=DataLoader(reserve()[0],batch_size=2,collate_fn=collate_detection)
    report={"candidate":evaluate(average,held)}
    model.transform.min_size=(384,)
    model.transform.max_size=640
    model.load_state_dict(src["model"])
    report["control"]=evaluate(model,held)
    (run/"reviewed-reserve.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
