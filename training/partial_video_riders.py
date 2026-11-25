"""Recover an unassigned rider whose lower body is outside a wide video frame."""
import numpy as np
from PIL import Image
import torch

from edge_riders import paired_seed, visible_pair
from helmet_flow import advance, fits_body
from photo_results import box_overlap
from predict import detector_objects
from temporal_head_search import verify


def claimed(body, head, people):
    return any(box_overlap(body, p["box"]) > .3 or fits_body(head, p["box"])
               or any(box_overlap(head, h["box"]) > .2 for h in p["cabezas"]) for p in people)


class PartialVideoRiders:
    def __init__(self, model, verifier, fps):
        self.model = model
        self.verifier = verifier
        self.interval = max(1, min(8, round(fps / 12))) if np.isfinite(fps) and fps > 0 else 3
        self.seeds = []
        self.previous = None

    @torch.inference_mode()
    def scan(self, image, people):
        tile_width = round(image.width / 4)
        tile_height = round(image.height / 4)
        top = image.height - tile_height
        pairs = []
        for left in np.linspace(0, image.width-tile_width, 7).round().astype(int):
            tile = image.crop((left, top, left+tile_width, image.height))
            for obj in detector_objects(*self.model[:3], tile, .40):
                if obj["class_id"] != 1:
                    continue
                body = (np.asarray(obj["box"]) + [left, top, left, top]).tolist()
                l, t, r, b = body
                if b < image.height - .08*tile_height or b-t < .4*tile_height:
                    continue
                if any(box_overlap(body, p["box"]) > .3 for p in people):
                    continue
                side = round(2.5 * max(b-t, r-l))
                x, y = round((l+r)/2-side/2), round(t+.18*(b-t)-.36*side)
                view = image.crop((x, y, x+side, y+side)).resize((256, 256), Image.Resampling.BILINEAR)
                options = []
                for proposal in detector_objects(*self.verifier.head_model[:3], view, .45):
                    if proposal["class_id"] != 2:
                        continue
                    head = (np.asarray(proposal["box"])*side/256 + [x,y,x,y]).tolist()
                    w, h = head[2]-head[0], head[3]-head[1]
                    if not fits_body(head, body) or not (.5 < w/max(h,1) < 1.7 and b-head[3] > .3*h):
                        continue
                    if claimed(body, head, people) or verify(image, head, self.verifier) < .90:
                        continue
                    options.append((obj["score"]*proposal["score"], body, head))
                if len(options) == 1:
                    pairs.append(options[0])
        result = []
        for _, body, head in sorted(pairs, key=lambda p: -p[0]):
            if not any(box_overlap(head, other) > .3 or box_overlap(body, torso) > .5 for torso,other in result):
                result.append((body, head))
        return result

    def find(self, image, people, frame):
        width = min(960, image.width)
        height = max(1, round(width*image.height/image.width))
        gray = np.asarray(image.convert("L").resize((width,height), Image.Resampling.BILINEAR))
        scale = np.array([width/image.width, height/image.height]*2)
        bounds = np.array([width,height]*2)
        active = []
        for item in self.seeds:
            anchor = item["anchor"]
            if (frame-anchor["frame"] > 4*self.interval or not advance(self.previous, gray, anchor["face"])
                    or not advance(self.previous, gray, anchor["torso"])):
                continue
            pair = visible_pair(gray, anchor, scale, bounds, require_edge=False)
            if pair is not None and not claimed(*pair, people):
                item["pair"] = pair
                active.append(item)
        if (frame-1) % self.interval == 0:
            for body, head in self.scan(image, people):
                matches = [item for item in active if box_overlap(head,item["pair"][1]) >= .45
                           and box_overlap(body,item["pair"][0]) >= .35]
                if len(matches) > 1 or (matches and matches[0]["anchor"]["frame"] == frame):
                    continue
                if matches:
                    # Fuse this localization with the independently tracked box in this frame.
                    head = np.mean([head, matches[0]["pair"][1]], axis=0).tolist()
                anchor = paired_seed(gray, image, head, body, scale, frame, segment=False)
                if anchor is None:
                    continue
                pair = visible_pair(gray, anchor, scale, bounds, require_edge=False)
                if pair is None:
                    continue
                if len(matches) == 1 and matches[0]["anchor"]["frame"] < frame:
                    item = matches[0]
                    # Evidence from overlapping tiles in one frame counts once.
                    item.update(anchor=anchor, pair=pair, checks=(item["checks"]+[frame])[-3:])
                elif not matches and not any(box_overlap(pair[1],item["pair"][1]) > .2 for item in active):
                    active.append(dict(anchor=anchor, pair=pair, checks=[frame]))
        self.previous = gray
        self.seeds = active
        return [(item["pair"], item["checks"]) for item in active if len(item["checks"]) == 3
                and not any(other is not item and box_overlap(other["pair"][1], item["pair"][1]) > .3
                            for other in active)]
