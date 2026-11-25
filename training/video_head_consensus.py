"""Cross-check a missing helmet in two rider-centered views of the current frame."""
import numpy as np
from PIL import Image
import torch

from edge_riders import paired_seed, visible_pair
from helmet_flow import advance, fits_body, redetect
from photo_results import box_overlap
from predict import detector_objects
from temporal_head_search import verify


@torch.inference_mode()
def helmet_candidates(image, body, model, specialist):
    left, top, right, bottom = body
    extent = max(right-left, bottom-top)
    center_x, head_y = (left+right)/2, top+.14*(bottom-top)
    width = min(image.width, max(32, round(extent * (1.6 if specialist else 2.4))))
    height = min(image.height, width if specialist else max(32, round(width*image.height/image.width)))
    x = min(max(0, round(center_x-width/2)), image.width-width)
    y = min(max(0, round(head_y-.28*height)), image.height-height)
    view = image.crop((x, y, x+width, y+height))
    if specialist:
        view = view.resize((256, 256), Image.Resampling.BILINEAR)
    boxes = []
    for item in detector_objects(*model[:3], view, .65):
        if item["class_id"] != 2:
            continue
        box = [v*s+offset for v,s,offset in zip(
            item["box"], (width/view.width, height/view.height)*2, (x,y)*2)]
        area = (box[2]-box[0])*(box[3]-box[1])
        if 0 < area <= .20*(right-left)*(bottom-top) and fits_body(box, body):
            boxes.append(box)
    return boxes


def agreed_head(image, body, model, verifier):
    first = helmet_candidates(image, body, model, False)
    if not first:
        return None
    second = helmet_candidates(image, body, verifier.head_model, True)
    pairs = [(a,b) for a in first for b in second if box_overlap(a,b) >= .55]
    if len(pairs) != 1:
        return None
    box = np.mean(pairs[0], axis=0).tolist()
    if verify(image, box, verifier) >= .93:
        return box
    return None


class VideoHeadConsensus:
    def __init__(self, model, verifier, edges, fps):
        self.model = model
        self.verifier = verifier
        self.edges = edges
        self.anchors = {}
        self.previous = None
        self.max_gap = min(24, max(3, round(.3*fps))) if np.isfinite(fps) and fps > 0 else 8

    def refine(self, image, people, ids, tracker, frame):
        # EdgeRiders has already processed this frame at tracking resolution.
        gray = self.edges.previous
        height, width = gray.shape
        scale = np.array([width/image.width, height/image.height]*2)
        bounds = np.array([width,height]*2)
        current = {ids[p["id"]]: p for p in people if p["id"] in ids}
        for key, anchor in list(self.anchors.items()):
            if (frame-anchor["frame"] > self.max_gap or not advance(self.previous, gray, anchor["face"])
                    or not advance(self.previous, gray, anchor["torso"])
                    or (key in current and current[key]["estado"] == "sin_casco")):
                del self.anchors[key]
        for key, rider in current.items():
            if not tracker.confirmed(key) or rider["cabezas"] or tracker.state(key) == "sin_casco":
                continue
            body = rider["box"]
            head = agreed_head(image, body, self.model, self.verifier)
            if head is None or any(other is not rider and fits_body(head, other["box"]) for other in people):
                continue
            track = tracker.items[key]
            track.setdefault("consensus_frames", []).append(frame)
            if tracker.state(key) == "con_casco":
                rider.update(estado="con_casco", cabezas=[dict(box=head, class_id=2, source="rider_consensus",
                                                             seed_frame=frame, verified_helmet=True)])
            anchor = paired_seed(gray, image, head, body, scale, frame)
            if anchor is not None and anchor["outline"] is not None:
                self.anchors[key] = anchor
        proposals = []
        for key in self.anchors.keys() | self.edges.anchors.keys():
            if key in current or frame-tracker.items[key]["last"] > self.max_gap:
                continue
            # Use only an established helmet track during a gap in body detections.
            if tracker.state(key) != "con_casco":
                continue
            candidates = [bank[key] for bank in (self.anchors, self.edges.anchors) if key in bank]
            for anchor in sorted(candidates, key=lambda a: a["frame"], reverse=True):
                if anchor["outline"] is None:
                    continue
                pair = visible_pair(gray, anchor, scale, bounds, require_edge=False)
                if pair is None:
                    continue
                body, head = pair
                if any(box_overlap(body, p["box"]) > .3 or fits_body(head,p["box"]) for p in people):
                    continue
                if not redetect(image, head, 2, self.verifier.head_model):
                    continue
                if verify(image, head, self.verifier) < .93:
                    # A tight moving outline may cut away the visor context.
                    # Reacquire the helmet with both detectors on the visible body.
                    detection = agreed_head(image, body, self.model, self.verifier)
                    if detection is None or box_overlap(detection, head) < .55:
                        continue
                proposals.append((key, body, head, anchor["frame"]))
                break
        for key, body, head, origin in proposals:
            if any(key != other and box_overlap(body, box) > .3 for other,box,_,_ in proposals):
                continue
            local = max((p["id"] for p in people), default=0)+1
            people.append(dict(id=local, box=body, moto=None, estado="con_casco", cabezas=[
                dict(box=head, class_id=2, source="rider_consensus", seed_frame=origin)]))
            ids[local] = key
        self.previous = gray
