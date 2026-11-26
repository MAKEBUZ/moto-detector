"""Fine-tune the local RPN and ROI box predictor on reviewed multi-rider scenes."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from data import SeparateHelmetDataset, collate_detection
from evaluate_paired_70_73 import reserve
from evaluate_rider_tracking import evaluate_temporal
from predict import load_model
from train_joint_clean import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontal-review", action="store_true")
    frontal = parser.parse_args().frontal_review
    run = "joint-v9-frontal-13" if frontal else "joint-crowded-riders-12"
    parent = "joint-occupant-negatives-11"
    torch.manual_seed(321)
    rng = random.Random(321)
    torch.set_num_threads(4)
    root = Path("data/prepared/helmet-separate-v2-expanded-09")
    if frontal:
        root = Path("data/prepared/frontal-scratch-01")
    train = SeparateHelmetDataset(root, "train", augment=False)
    validation = DataLoader(SeparateHelmetDataset(root, "val", augment=False), batch_size=2, collate_fn=collate_detection)
    reserved, _ = reserve()
    reserved = DataLoader(reserved, batch_size=2, collate_fn=collate_detection)
    manifests = list(validation.dataset.rows)
    manifests += [json.loads(line) for line in Path("data/prepared/helmet-separate-v2-test-02/test-manifest.jsonl").read_text().splitlines()]
    for field in ("sha256", "group_id"):
        assert not ({r[field] for r in train.rows} & {r[field] for r in manifests})
    crowded = []
    for i, row in enumerate(train.rows):
        bikes = defaultdict(set)
        for link in row["links"]:
            if link.get("motorcycle_id") and link["status"] == "confirmed":
                bikes[link["motorcycle_id"]].add(link["person_id"])
        if any(len(ids) >= 2 for ids in bikes.values()):
            crowded.append(i)
    reviewed = []
    if frontal:
        hashes = set()
        for batch in ("ocupantes-solapados-01", "visores-frontales-01"):
            path = Path("data/annotations") / batch / f"{batch}-reviewed.json"
            hashes.update(s["sha256"] for s in json.loads(path.read_text(encoding="utf-8"))["scenes"])
        reviewed = [i for i, r in enumerate(train.rows) if r["sha256"] in hashes]
        assert len(reviewed) == len(hashes) == 16
        locked = json.loads(Path("data/annotations/reserva-frontal-01/reserva.json").read_text())
        assert not set(locked["group_ids"]) & {r["group_id"] for r in train.rows}
        replay = [i for i in range(len(train)) if i not in reviewed]
    assert crowded
    folder = Path("runs") / run
    folder.mkdir(exist_ok=False)
    model, cfg, *_ = load_model(parent)
    baseline = evaluate(model, validation)
    reference = evaluate(model, reserved)
    model.requires_grad_(False)
    modules = (model.roi_heads.box_predictor,) if frontal else (model.rpn.head, model.roi_heads.box_predictor)
    for m in modules:
        m.requires_grad_(True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=.000004 if frontal else .000025)
    cfg = dict(cfg, run=run, parent_run=parent, initialization="crowded_rider_transfer",
               parent_sha256=hashlib.sha256((Path("runs")/parent/"best.pt").read_bytes()).hexdigest())
    history = []
    best = None
    for step in range(1, 65 if frontal else 61):
        model.train()
        model.backbone.eval()
        indices = [rng.choice(crowded), rng.randrange(len(train))]
        if frontal:
            if (step-1) % len(reviewed) == 0:
                rng.shuffle(reviewed)
            indices = [reviewed[(step-1) % len(reviewed)], rng.choice(replay)]
        images, labels = collate_detection([train[i] for i in indices])
        losses = model([x.cuda() for x in images], [{k:v.cuda() for k,v in y.items()} for y in labels])
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % (16 if frontal else 10):
            continue
        val = evaluate(model, validation)
        c = val["counts"][1]
        rank = (2*c["tp"]/(2*c["tp"]+c["fp"]+c["fn"]), val["metrics"]["map50"])
        history.append(dict(step=step, loss=loss.item(), validation=val))
        if best is None or rank > best:
            best = rank
            best_step = step
            torch.save(dict(model=model.state_dict(), config=cfg, epoch=step), folder/"best.pt")
        print(step, rank, flush=True)
    chosen, *_ = load_model(run)
    out = Path("reports/training") / run
    out.mkdir(exist_ok=True)
    report = dict(parent=parent, train_images=len(train), crowded=[train.rows[i]["image_id"] for i in crowded],
                  history=history, baseline=baseline, reserved_baseline=reference, reserved_candidate=evaluate(chosen, reserved))
    if frontal:
        report.update(selected_step=best_step, learning_rate=.000004, seed=321, parent_sha256=cfg["parent_sha256"])
        report["reviewed_images"] = [train.rows[i]["image_id"] for i in reviewed]
    (out/"training.json").write_text(json.dumps(report, indent=2))
    evaluate_temporal(Path("reports/training/tracking-occupants-v9/video.json"), candidate_run=run)


if __name__ == "__main__":
    main()
