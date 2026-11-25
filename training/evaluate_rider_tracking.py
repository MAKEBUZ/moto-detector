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

    stream = Stream(prior["source"], run)
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


def evaluate_temporal(control_path):
    control = read(control_path)
    baseline = read(Path("reports/training/tracking-occupants-v4/comparison.json"))
    marks = read(Path("runs") / control["run"] / "paired-review.json")["review"]
    assert control["tracking_version"] == 4 and baseline["source"] == Path(control["source"]).name
    assert control["run"] == baseline["run"] and control["counts"] == baseline["corrected_counts"]
    stream = Stream(control["source"], control["run"])
    rows = []
    added = []
    try:
        refine = stream.temporal_heads.refine

        def capture(image, people, ids, tracker, frame):
            refine(image, people, ids, tracker, frame)
            for rider in people:
                for head in rider["cabezas"]:
                    if head.get("source") == "temporal_patch_check":
                        added.append(dict(frame=frame, key=ids[rider["id"]], box=head["box"], score=head["score"]))
            for mark in marks:
                if mark["frame"] == frame:
                    rows.append(dict(frame=frame, category=mark["category"], review_id=mark["track"],
                                     **match(mark, people, ids)))

        stream.temporal_heads.refine = capture
        while stream.next() is not None:
            pass
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
    report = dict(model=control["run"], source=baseline["source"], frames=video["frames"],
                  control=baseline["totals"]["corrected"], candidate=scores,
                  control_counts=control["counts"], candidate_counts=video["counts"], marks=rows)
    output = Path("reports/training/tracking-occupants-v5")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("comparison", report), ("video", video), ("candidates", added)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(scores, video["counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", type=Path, metavar="VIDEO_JSON", help="compare with a saved v4 video report")
    args = parser.parse_args()
    if args.temporal:
        evaluate_temporal(args.temporal)
    else:
        main()
