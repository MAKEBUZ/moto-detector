"""Keep an already counted helmeted rider visible while they leave the frame."""
import cv2
import numpy as np
from PIL import Image

from helmet_flow import advance, seed
from photo_results import box_overlap


def helmet_outline(image, box):
    """Tighten a full detector box before part of the helmet becomes invisible."""
    x, y, r, b = box
    w, h = r-x, b-y
    tile = image.crop((round(x-.25*w), round(y-.25*h), round(r+.25*w), round(b+.25*h)))
    rgb = np.asarray(tile.resize((144, 144), Image.Resampling.BILINEAR))
    mask = np.zeros((144, 144), np.uint8)
    cv2.grabCut(rgb, mask, (24, 24, 96, 96), None, None, 3, cv2.GC_INIT_WITH_RECT)
    contours = cv2.findContours(mask % 2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < .25*96**2:
        return None
    l, t, width, height = cv2.boundingRect(largest)
    return (np.array([l, t, l+width, t+height]) - 24)/96


def paired_seed(gray, image, head, body, scale, frame, *, segment=True):
    head = np.asarray(head)*scale
    body = np.asarray(body)*scale
    x, y, right, bottom = head
    left, top, body_right, body_bottom = body
    if min(x, y, gray.shape[1] - right, gray.shape[0] - bottom) < 2:
        return None
    face = seed(gray, head, frame)
    if face is None:
        return None
    # Select the shoulders and chest, excluding features inside the helmet.
    w, h = right - x, bottom - y
    torso_box = np.array([max(left, x - .45*w), max(top, y + .58*h),
                          min(body_right, right + .45*w), min(body_bottom, bottom + .85*h)])
    if min(torso_box[2:] - torso_box[:2]) < 8:
        return None
    left, top, r, b = np.rint(torso_box).astype(int)
    region = gray[top:b, left:r]
    mask = np.full(region.shape, 255, np.uint8)
    cv2.rectangle(mask, (int(x) - left - 1, int(y) - top - 1),
                  (int(right) - left + 1, int(bottom) - top + 1), 0, -1)
    points = cv2.goodFeaturesToTrack(region, 100, .01, 2.5, mask=mask)
    if points is None or len(points) < 6:
        return None
    torso = dict(box=torso_box, points=points + np.float32([left, top]))
    for part in (face, torso):
        part["origins"] = part["points"].copy()
    # Keep only a local patch of the original frame for the appearance check.
    left, top = np.floor(np.minimum(head[:2], torso_box[:2]) - 5).astype(int).clip(0)
    right, bottom = np.ceil(np.maximum(head[2:], torso_box[2:]) + 5).astype(int)
    return dict(face=face, torso=torso, torso_origin=torso_box.copy(), initial=head,
                body=body, frame=frame, patch=gray[top:bottom, left:right].copy(),
                offset=np.array([left, top]),
                outline=helmet_outline(image, head/scale) if segment else np.array([0., 0., 1., 1.]))


def same_features(gray, region, anchor):
    scores = []
    for source, dest in zip(region["origins"][:, 0], region["points"][:, 0]):
        x, y = np.rint(source - anchor["offset"]).astype(int)
        u, v = np.rint(dest).astype(int)
        if min(u, v, x, y) < 4 or u+4 >= gray.shape[1] or v+4 >= gray.shape[0]:
            continue
        before = anchor["patch"][y-4:y+5,x-4:x+5]
        after = gray[v-4:v+5,u-4:u+5]
        if before.shape == after.shape and min(before.std(), after.std()) >= 8:
            scores.append(cv2.matchTemplate(after, before, cv2.TM_CCOEFF_NORMED).item())
    # A clipped helmet can lose most of its surface. Require either consistent
    # agreement across the visible points or six individually strong matches.
    return len(scores) >= 3 and (np.median(scores) >= .80 or np.count_nonzero(np.asarray(scores) >= .85) >= 6)


def visible_pair(gray, anchor, scale, bounds, *, require_edge=True):
    face = anchor["face"]["box"]
    size = face[2:] - face[:2]
    initial = anchor["initial"]
    magnification = np.tile(size / (initial[2:] - initial[:2]), 2)
    origin = np.tile(initial[:2], 2)
    position = np.tile(face[:2], 2)
    body = position + (anchor["body"] - origin) * magnification
    clipped = np.clip(body, 0, bounds)
    if (require_edge and np.array_equal(body, clipped)) or min(clipped[2:] - clipped[:2]) < 8:
        return None
    # Independent motion and appearance checks on the head and upper body.
    expected = position + (anchor["torso_origin"] - origin) * magnification
    if box_overlap(expected, anchor["torso"]["box"]) < .45:
        return None
    if not all(same_features(gray, anchor[part], anchor) for part in ("face", "torso")):
        return None
    outline = position + anchor["outline"] * np.tile(size, 2)
    helmet = np.clip(outline, 0, bounds)
    visible_size = helmet[2:] - helmet[:2]
    if min(visible_size) < 5 or np.prod(visible_size) < .35 * np.prod(outline[2:] - outline[:2]):
        return None
    return (clipped / scale).tolist(), (helmet / scale).tolist()


class EdgeRiders:
    def __init__(self, fps):
        self.max_age = round(1.25*fps) if np.isfinite(fps) and fps > 0 else 30
        self.max_age = min(90, max(3, self.max_age))
        self.previous = None
        self.anchors = {}

    def refine(self, image, people, ids, tracker, frame):
        width = min(960, image.width)
        height = max(1, round(width*image.height/image.width))
        gray = np.asarray(image.convert("L").resize((width, height), Image.Resampling.BILINEAR))
        scale = np.array([width/image.width, height/image.height] * 2)
        bounds = np.array([width, height] * 2)
        current = {ids[p["id"]]: p for p in people if p["id"] in ids}
        proposals = []
        for key, anchor in list(self.anchors.items()):
            if (frame - anchor["frame"] > self.max_age or not advance(self.previous, gray, anchor["face"])
                    or not advance(self.previous, gray, anchor["torso"])):
                del self.anchors[key]
                continue
            if key in current:
                if current[key]["estado"] == "sin_casco":
                    del self.anchors[key]
                continue
            if tracker.detector_state(key) != "con_casco" or anchor["outline"] is None:
                continue
            boxes = visible_pair(gray, anchor, scale, bounds)
            if boxes is None:
                continue
            body, helmet = boxes
            if any(box_overlap(body, p["box"]) > .3 or any(box_overlap(helmet, h["box"]) > .2
                   for h in p["cabezas"]) for p in people):
                continue
            proposals.append((key, body, helmet, anchor["frame"]))

        for key, body, helmet, origin in proposals:
            if any(other != key and box_overlap(body, box) > .3 for other, box, _, _ in proposals):
                continue
            local = max((p["id"] for p in people), default=0) + 1
            head = dict(box=helmet, class_id=2, source="edge_flow", seed_frame=origin)
            people.append(dict(id=local, box=body, moto=None, estado="con_casco", cabezas=[head]))
            ids[local] = key
            track = tracker.items[key]
            track["edge_returns"] = track.get("edge_returns", 0) + 1
        for key, p in current.items():
            if not tracker.confirmed(key) or p["estado"] != "con_casco" or len(p["cabezas"]) != 1:
                continue
            head = p["cabezas"][0]
            if head.get("verified_helmet"):
                anchor = paired_seed(gray, image, head["box"], p["box"], scale, frame)
                if anchor is not None:
                    self.anchors[key] = anchor
        self.previous = gray
