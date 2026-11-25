"""Compare the previous and current trackers on identical video detections and reviewed regions."""
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from photo_results import summarize
from traffic_video import RiderTracks, Stream, Tracks, overlap


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def match(review, riders, ids):
    helmet = review["category"] == "con_casco"
    best_iou, key = 0., None
    for rider in riders:
        if rider["id"] not in ids:
            continue
        boxes = ([h["box"] for h in rider["cabezas"] if h["class_id"] == 2]
                 if helmet else [rider["box"]])
        for box in boxes:
            score = overlap(box, review["reference"])
            if score > best_iou:
                best_iou, key = score, ids[rider["id"]]
    return dict(iou=best_iou, track=key if best_iou >= .5 else None)


def compare_cached_default(by_frame, output):
    cache = Path("reports/training/wide-zero-heads/baseline.jsonl")
    prior = read(Path("reports/training/wide-temporal-check/video.json"))
    report = dict(run=prior["run"], source=Path(prior["source"]).name, input=cache.as_posix())
    trackers = {"original": Tracks(), "corrected": RiderTracks()}
    matched = defaultdict(list)
    with cache.open(encoding="utf-8") as frames:
        for n, line in enumerate(frames, 1):
            row = json.loads(line)
            assert row["frame"] == n
            people = summarize(row["objects"])["ocupantes"]
            for name, tracker in trackers.items():
                ids = tracker.update(people, n)
                for mark in by_frame.get(n, []):
                    key = match(mark, people, ids)["track"]
                    matched[name].append((mark["category"], key))
    assert n == prior["frames"] and trackers["original"].count() == prior["counts"]["ocupantes"]
    report["frames"] = n
    for name, tracker in trackers.items():
        assert len(matched[name]) == 17
        confirmed = [(c, k) for c, k in matched[name] if tracker.confirmed(k)]
        report[name] = dict(occupants=tracker.count(), **tracker.state_counts(),
                            helmet_final=sum(c == "con_casco" and tracker.state(k) == "con_casco"
                                             for c, k in confirmed),
                            false_final=sum(c == "no_persona" for c, k in confirmed))
    (output / "default-model.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    run = "joint-occupant-negatives-11"
    root = Path("runs") / run
    prior = read(root / "video.json")
    marks = read(root / "paired-review.json")["review"]
    by_frame = defaultdict(list)
    for mark in marks:
        by_frame[mark["frame"]].append(mark)

    stream = Stream(prior["source"], run, motion_check=False)
    stream.temporal_heads = None  # Preserve the version 4 benchmark when replayed.
    old = Tracks()
    update = stream.people.update
    results = []

    def compare(riders, frame):
        before = old.update(riders, frame)
        after = update(riders, frame)
        for mark in by_frame[frame]:
            control = match(mark, riders, before)
            assert (control["track"] is not None) == mark["detected"]
            if mark["detected"]:
                assert control["track"] == mark["candidate_track"]
            results.append(dict(frame=frame, review_id=mark["track"], category=mark["category"],
                                original=control, corrected=match(mark, riders, after)))
        return after

    stream.people.update = compare
    try:
        while stream.next() is not None:
            pass
        with TemporaryDirectory() as tmp:
            video = read(stream.save(Path(tmp), "completo") / "video.json")
    finally:
        stream.cap.release()
    assert video["frames"] == prior["frames"] == 1500 and len(results) == len(marks) == 17
    assert old.state_counts() == {k: prior["counts"][k] for k in old.state_counts()}
    assert old.count() == prior["counts"]["ocupantes"]
    totals = {}
    for name, tracker in (("original", old), ("corrected", stream.people)):
        counts = dict(helmet_at_frame=0, helmet_final=0, false_at_frame=0, false_final=0)
        confirmed_helmet_ids = set()
        for row in results:
            target = row[name]
            key = target["track"]
            target["confirmed"] = tracker.confirmed(key)
            target["final_state"] = tracker.state(key) if target["confirmed"] else None
            if key is None:
                continue
            if row["category"] == "con_casco":
                counts["helmet_at_frame"] += 1
                if target["final_state"] == "con_casco":
                    counts["helmet_final"] += 1
                    confirmed_helmet_ids.add(key)
            else:
                counts["false_at_frame"] += 1
                counts["false_final"] += int(target["confirmed"])
        totals[name] = dict(**counts, distinct_confirmed_helmet_tracks=len(confirmed_helmet_ids))

    output = Path("reports/training/tracking-occupants-v4")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(run=run, source=Path(prior["source"]).name, frames=video["frames"],
                  original_version=prior["tracking_version"], corrected_version=video["tracking_version"],
                  criterion="IoU >= 0.5 at the marked frame; confirmed at end of video; helmet requires final con_casco",
                  totals=totals, original_counts=prior["counts"], corrected_counts=video["counts"], marks=results)
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "video.json").write_text(json.dumps(video, indent=2), encoding="utf-8")
    compare_cached_default(by_frame, output)
    print(totals, video["counts"], flush=True)


def evaluate_temporal(control_path, motion=False, edge=False, consensus=False, partial=False, candidate_run=None):
    control = read(control_path)
    if candidate_run:
        version, source = 9, "partial_rider"
    elif partial:
        version, source = 8, "partial_rider"
    elif consensus:
        version, source = 7, "rider_consensus"
    elif edge:
        version, source = 6, "edge_flow"
    elif motion:
        version, source = 5, "helmet_flow"
    else:
        version, source = 4, "temporal_patch_check"
    label = candidate_run or f"v{version+1}"
    reference_dir = Path(f"reports/training/tracking-occupants-v{version}")
    baseline = read(reference_dir / "comparison.json")
    marks = read(Path("runs") / control["run"] / "paired-review.json")["review"]
    assert control["tracking_version"] == version and baseline["source"] == Path(control["source"]).name
    assert control["run"] == baseline.get("run", baseline.get("model"))
    assert control["counts"] == baseline.get("corrected_counts", baseline.get("candidate_counts"))
    stream = Stream(control["source"], control["run"], motion_check=version >= 5,
                    edge_check=version >= 6, consensus_check=version >= 7, partial_check=partial or bool(candidate_run), detector_run=candidate_run)
    rows = []
    added = []
    extra_frames = read(reference_dir / "additional-frames.json") if version >= 5 else None
    try:
        search = {4: stream.temporal_heads, 5: stream.motion_heads, 6: stream.edge_riders,
                  7: stream.rider_heads, 8: stream.rider_heads, 9: stream.rider_heads}[version]
        refine = search.refine

        def capture(image, people, ids, tracker, frame):
            refine(image, people, ids, tracker, frame)
            for rider in people:
                for head in rider["cabezas"]:
                    if head.get("source") == source:
                        evidence = dict(frame=frame, key=ids[rider["id"]], box=head["box"], seed_frame=head["seed_frame"])
                        if version >= 6:
                            evidence["body"] = rider["box"]
                        else:
                            evidence["score"] = head["score"]
                        added.append(evidence)
            if extra_frames is not None and frame in extra_frames["frames"]:
                # Match distinct riders to distinct visible helmets in each annotated frame.
                remaining = dict(ids)
                for row in extra_frames["rows"]:
                    if row["frame"] == frame:
                        sx = image.width/extra_frames["coordinate_space"][0]
                        sy = image.height/extra_frames["coordinate_space"][1]
                        box = [v*s for v,s in zip(row["box"], (sx,sy,sx,sy))]
                        row[label] = match(dict(category="con_casco", reference=box), people, remaining)
                        key = row[label]["track"]
                        remaining = {i:k for i,k in remaining.items() if k != key}
            for mark in marks:
                if mark["frame"] == frame:
                    rows.append(dict(frame=frame, category=mark["category"], review_id=mark["track"],
                                     **match(mark, people, ids)))

        search.refine = capture
        while stream.next() is not None:
            if stream.frames % 250 == 0:
                print(f"{stream.frames} frames", flush=True)
        with TemporaryDirectory() as tmp:
            video = read(stream.save(Path(tmp), "completo") / "video.json")
    finally:
        stream.cap.release()
    assert video["frames"] == control["frames"] == 1500 and len(rows) == len(marks) == 17
    for row in rows:
        key = row["track"]
        row["confirmed"] = stream.people.confirmed(key)
        row["final_state"] = stream.people.state(key) if row["confirmed"] else None
    helmets = [r for r in rows if r["category"] == "con_casco"]
    false_people = [r for r in rows if r["category"] == "no_persona"]
    scores = dict(helmet_at_frame=sum(r["track"] is not None for r in helmets),
                  helmet_final=sum(r["final_state"] == "con_casco" for r in helmets),
                  false_at_frame=sum(r["track"] is not None for r in false_people),
                  false_final=sum(r["confirmed"] for r in false_people),
                  distinct_confirmed_helmet_tracks=len({r["track"] for r in helmets if r["final_state"] == "con_casco"}))
    report = dict(model=candidate_run or control["run"], source=baseline["source"], frames=video["frames"],
                  control=baseline["candidate"] if version >= 5 else baseline["totals"]["corrected"], candidate=scores,
                  control_counts=control["counts"], candidate_counts=video["counts"], marks=rows)
    output = Path("reports/training") / (candidate_run or f"tracking-occupants-v{version+1}")
    output.mkdir(parents=True, exist_ok=True)
    if extra_frames is not None:
        for row in extra_frames["rows"]:
            key = row[label]["track"]
            row[label]["final_state"] = stream.people.state(key) if stream.people.confirmed(key) else None
        matches = [r[label]["track"] for r in extra_frames["rows"] if r[label]["track"] is not None]
        extra_frames["scores"][label] = dict(visible=len(extra_frames["rows"]), at_frame=len(matches),
            final=sum(stream.people.confirmed(k) and stream.people.state(k) == "con_casco" for k in matches))
        (output / "additional-frames.json").write_text(json.dumps(extra_frames, indent=2), encoding="utf-8")
        report["false_regions_with_final_helmet"] = sum(r["final_state"] == "con_casco" for r in false_people)
    for name, data in (("comparison", report), ("video", video), ("candidates", added)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(scores, video["counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--temporal", type=Path, metavar="VIDEO_JSON", help="compare with a saved v4 video report")
    modes.add_argument("--motion", type=Path, metavar="VIDEO_JSON", help="compare with a saved v5 video report")
    modes.add_argument("--edge", type=Path, metavar="VIDEO_JSON", help="compare with a saved v6 video report")
    modes.add_argument("--consensus", type=Path, metavar="VIDEO_JSON", help="compare with a saved v7 video report")
    modes.add_argument("--partial", type=Path, metavar="VIDEO_JSON", help="compare with a saved v8 video report")
    args = parser.parse_args()
    if args.partial:
        evaluate_temporal(args.partial, partial=True)
    elif args.consensus:
        evaluate_temporal(args.consensus, consensus=True)
    elif args.edge:
        evaluate_temporal(args.edge, edge=True)
    elif args.motion:
        evaluate_temporal(args.motion, motion=True)
    elif args.temporal:
        evaluate_temporal(args.temporal)
    else:
        main()
