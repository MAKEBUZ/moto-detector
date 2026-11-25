"""Follow a directly verified helmet between detections, using current-frame pixels."""
import cv2
import numpy as np
from PIL import Image
import torch

from photo_results import box_overlap
from predict import detector_objects
from temporal_head_search import verify


def crop_signature(gray, box):
    x, y, r, b = np.rint(box).astype(int)
    if x < 0 or y < 0 or r > gray.shape[1] or b > gray.shape[0] or min(r-x, b-y) < 8:
        return None
    return cv2.resize(gray[y:b, x:r], (24, 24), interpolation=cv2.INTER_AREA)


def seed(gray, box, frame):
    signature = crop_signature(gray, box)
    if signature is None or signature.std() < 10:
        return None
    x, y, r, b = np.rint(box).astype(int)
    points = cv2.goodFeaturesToTrack(gray[y:b, x:r], 40, .015, 3)
    if points is None or len(points) < 6:
        return None
    return dict(box=box, points=points + np.float32([x, y]), source=frame, signature=signature)


def advance(previous, gray, anchor):
    points = anchor["points"]
    forward, valid, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
    if forward is None:
        return False
    back, back_valid, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None)
    if back is None:
        return False
    keep = valid.ravel().astype(bool) & back_valid.ravel().astype(bool)
    keep &= np.linalg.norm((points-back)[:, 0], axis=1) < 1.2
    old, new = points[keep, 0], forward[keep, 0]
    if len(old) < 6:
        return False
    matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=1.8)
    if matrix is None or inliers.sum() < max(6, .7*len(old)):
        return False
    scale = np.linalg.norm(matrix[0, :2])
    if not .87 <= scale <= 1.15:
        return False
    box = anchor["box"]
    center = matrix[:, :2] @ ((box[:2]+box[2:])/2) + matrix[:, 2]
    half = (box[2:]-box[:2]) * scale/2
    if "origins" in anchor:
        anchor["origins"] = anchor["origins"][keep][inliers.ravel().astype(bool)]
    anchor.update(box=np.r_[center-half, center+half], points=new[inliers.ravel().astype(bool), None, :])
    return True


def fits_body(box, body):
    x,y,X,Y = body
    cx,cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    return x <= cx <= X and y-.24*(Y-y) <= cy <= y+.55*(Y-y)


@torch.inference_mode()
def redetect(image, box, category, specialist):
    """Localize the candidate again in the 256px view used by the rider-head specialist."""
    l,t,r,b = box
    extent = max(32., 2.8*max(r-l,b-t))
    side = round(extent)
    cx,cy = (l+r)/2, (t+b)/2
    x, y = round(cx-extent/2), round(cy-.4*extent)
    view = image.crop((x,y,x+side,y+side)).resize((256,256), Image.Resampling.BILINEAR)
    target = [(l-x)*256/side,(t-y)*256/side,(r-x)*256/side,(b-y)*256/side]
    return any(o["class_id"] == category and box_overlap(o["box"], target) >= .5
               for o in detector_objects(*specialist[:3], view, .5))


class HelmetFlow:
    def __init__(self, verifier, fps):
        self.verifier = verifier
        self.max_age = max(3, min(36, round(.5 * fps))) if np.isfinite(fps) else 15
        self.previous = None
        self.anchors = {}

    def refine(self, image, people, ids, tracker, frame):
        width = min(960, image.width)
        height = max(1, round(width*image.height/image.width))
        gray = np.asarray(image.convert("L").resize((width, height), Image.Resampling.BILINEAR))
        scale = np.array([width/image.width, height/image.height]*2)
        current = {ids[p["id"]]: p for p in people if p["id"] in ids}
        occupied = [h["box"] for p in people for h in p["cabezas"]]
        for key, rider in current.items():
            if rider["estado"] == "sin_casco" and redetect(image, rider["cabezas"][0]["box"], 3, self.verifier.head_model):
                track = tracker.items[key]
                track.setdefault("first_context_bare", frame)
                track["last_context_bare"] = frame
        for key, anchor in list(self.anchors.items()):
            if frame-anchor["source"] > self.max_age or not advance(self.previous, gray, anchor):
                del self.anchors[key]
                continue
            rider = current.get(key)
            if rider is not None and rider["cabezas"]:
                if (rider["estado"] == "sin_casco" and tracker.items[key].get("last_context_bare") == frame
                        and box_overlap((anchor["box"]/scale).tolist(), rider["cabezas"][0]["box"]) >= .5):
                    del self.anchors[key]
                continue
            if not tracker.confirmed(key):
                continue
            box = (anchor["box"]/scale).tolist()
            if any(box_overlap(box, other) > .25 for other in occupied):
                continue
            if rider is None:
                track = tracker.items[key]
                if frame-track["last"] > 3:
                    continue
                # Only reconnect an actual detector body to an already counted rider.
                options = [p for p in people if p["id"] not in ids and not p["cabezas"]
                           and fits_body(box, p["box"]) and box_overlap(p["box"], track["box"]) >= .65]
                if len(options) != 1:
                    continue
                rider = options[0]
            l,t,r,b = box
            x,y,X,Y = rider["box"]
            # Rounded 95th percentile of head/body area on 139 confirmed training pairs.
            if not fits_body(box, rider["box"]) or (r-l)*(b-t) > .20*(X-x)*(Y-y):
                continue
            patch = crop_signature(gray, anchor["box"])
            if patch is None or patch.std() < 10:
                continue
            similarity = cv2.matchTemplate(patch, anchor["signature"], cv2.TM_CCOEFF_NORMED).item()
            if similarity < .78:
                continue
            confidence = verify(image, box, self.verifier)
            if confidence < .9 or not redetect(image, box, 2, self.verifier.head_model):
                continue
            ids[rider["id"]] = key
            rider["estado"] = "con_casco"
            rider["cabezas"] = [dict(box=box, class_id=2, score=confidence,
                                     source="helmet_flow", seed_frame=anchor["source"])]
            occupied.append(box)
            track = tracker.items[key]
            if key in current:
                track["flow_checks"] = track.get("flow_checks", 0) + 1
                track.setdefault("flow_anchors", set()).add(anchor["source"])
            else:
                track["flow_returns"] = track.get("flow_returns", 0) + 1
        for key, rider in current.items():
            if rider["estado"] == "con_casco" and rider["cabezas"][0].get("verified_helmet"):
                anchor = seed(gray, np.asarray(rider["cabezas"][0]["box"])*scale, frame)
                if anchor is not None:
                    self.anchors[key] = anchor
        self.previous = gray
