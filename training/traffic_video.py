"""Local video analysis, short-term tracking and queries over saved analyses."""
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from photo_results import annotate, summarize
from predict import analyze_image, load_model


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area = lambda box: (box[2]-box[0]) * (box[3]-box[1])
    return intersection / max(1, area(a) + area(b) - intersection)


class Tracks:
    def __init__(self):
        self.items = {}

    def match_score(self, track, obj, gap):
        return overlap(track["box"], obj["box"])

    def observe(self, track, obj, frame):
        track.update(last=frame, box=obj["box"])

    def update(self, objects, frame):
        candidates = []
        for key, track in self.items.items():
            gap = frame - track["last"]
            if gap <= 8 or (track["hits"] >= 3 and gap <= 24):
                for index, obj in enumerate(objects):
                    score = self.match_score(track, obj, gap)
                    if score >= (.3 if gap <= 8 else .65):
                        candidates.append((gap <= 8, score, key, index))
        assigned, used = {}, set()
        for _, _, key, index in sorted(candidates, reverse=True):
            if key not in used and index not in assigned:
                assigned[index] = key
                used.add(key)
        mapping = {}
        for index, obj in enumerate(objects):
            key = assigned.get(index, len(self.items)+1)
            track = self.items.setdefault(key, dict(first=frame, hits=0, votes=Counter()))
            track["hits"] += 1
            self.observe(track, obj, frame)
            if obj.get("estado") in ("con_casco", "sin_casco"):
                track["votes"][obj["estado"]] += 1
            if obj.get("estado") == "con_casco" and any(h.get("verified_helmet") for h in obj.get("cabezas", [])):
                frames = track.setdefault("helmet_checks", [])
                if len(frames) < 3 and (not frames or frame - frames[-1] >= 5):
                    frames.append(frame)
            mapping[obj["id"]] = key
        return mapping

    def confirmed(self, key):
        return key is not None and self.items[key]["hits"] >= 3

    def count(self):
        return sum(self.confirmed(key) for key in self.items)

    def state(self, key):
        track = self.items[key]
        votes = track["votes"]
        for state, count in votes.items():
            if count >= 3 and count >= .5 * track["hits"] and count >= .8 * sum(votes.values()):
                return state
        if (len(track.get("helmet_checks", [])) == 3 and votes["sin_casco"] == 0
                and votes["con_casco"] >= max(6, .2 * track["hits"])):
            return "con_casco"
        return "indeterminado"

    def state_counts(self):
        states = dict(con_casco=0, sin_casco=0, indeterminado=0)
        for key in self.items:
            if self.confirmed(key):
                states[self.state(key)] += 1
        return states

    def evidence(self):
        rows = []
        for key, t in self.items.items():
            if not self.confirmed(key):
                continue
            row = dict(id=key, primer_fotograma=t["first"], ultimo_fotograma=t["last"],
                 detecciones=t["hits"], votos=dict(t["votes"]), estado=self.state(key),
                 fotogramas_casco_verificado=t.get("helmet_checks", []))
            if t.get("patch_checks"):
                row["comprobaciones_temporales_de_casco"] = t["patch_checks"]
            if t.get("flow_checks"):
                row["comprobaciones_por_movimiento"] = t["flow_checks"]
                row["origenes_por_movimiento"] = sorted(t["flow_anchors"])
            if t.get("flow_returns"):
                row["reasociaciones_por_movimiento"] = t["flow_returns"]
            if t.get("edge_returns"):
                row["reapariciones_parciales_en_borde"] = t["edge_returns"]
            if t.get("partial_checks"):
                row["pares_casco_torso_en_borde_inferior"] = sorted(t["partial_checks"])
            if t.get("consensus_frames"):
                row["fotogramas_con_doble_localizacion"] = t["consensus_frames"]
            if "first_context_bare" in t:
                row["intervalo_sin_casco_relocalizado"] = [t["first_context_bare"], t["last_context_bare"]]
            if row["estado"] == "indeterminado":
                total = sum(t["votes"].values())
                strongest = max(t["votes"].values(), default=0)
                if total == 0:
                    reasons = ["sin_votos_de_cabeza"]
                elif strongest < 3:
                    reasons = ["votos_insuficientes"]
                else:
                    reasons = []
                    if strongest < .5 * t["hits"]:
                        reasons.append("cobertura_insuficiente")
                    if strongest < .8 * total:
                        reasons.append("votos_contradictorios")
                    if not reasons and "first_bare" in t and t["last_bare"] - t["first_bare"] < 5:
                        reasons.append("votos_sin_casco_demasiado_proximos")
                row["motivos"] = reasons
            rows.append(row)
        return rows


class MotorcycleTracks(Tracks):
    """Associate moving bikes in portrait video through short detection gaps."""

    def update(self, objects, frame):
        def area(obj):
            l,t,r,b = obj["box"]
            return (r-l)*(b-t)

        kept, aliases = [], {}
        for obj in sorted(objects, key=area, reverse=True):
            x,y,X,Y = obj["box"]
            for other in kept:
                if obj.get("ocupantes") and other.get("ocupantes"):
                    continue
                l,t,r,b = other["box"]
                intersection = max(0,min(X,r)-max(x,l))*max(0,min(Y,b)-max(y,t))
                if (intersection >= .65*area(obj) and abs(x+X-l-r) <= .45*(r-l)
                        and abs(y+Y-t-b) <= .65*(b-t)):
                    aliases[obj["id"]] = other["id"]
                    break
            else:
                kept.append(obj)
        mapping = super().update(kept, frame)
        mapping.update({key: mapping[parent] for key,parent in aliases.items()})
        return mapping

    def observe(self, track, obj, frame):
        if "box" in track:
            l,t,r,b = track["box"]
            x,y,X,Y = obj["box"]
            elapsed = frame-track["last"]
            vx,vy = track.get("velocity", (0., 0.))
            track["velocity"] = (.4*vx + .6*(x+X-l-r)/(2*elapsed),
                                 .4*vy + .6*(y+Y-t-b)/(2*elapsed))
        super().observe(track, obj, frame)

    def match_score(self, track, obj, gap):
        l,t,r,b = track["box"]
        vx,vy = track.get("velocity", (0., 0.))
        dx,dy = vx*min(gap, 8), vy*min(gap, 8)
        predicted = [l+dx, t+dy, r+dx, b+dy]
        score = overlap(predicted, obj["box"])
        if gap > 8 and track["hits"] >= 3:
            if abs(vx) < .05*(r-l) and abs(vy) < .05*(b-t):
                dx = dy = 0.
                score = max(score, overlap(track["box"], obj["box"]))
            # Long gaps need stronger evidence. Permit moderate scale growth
            # when the new center agrees with the predicted position.
            x,y,X,Y = obj["box"]
            if (abs(x+X-l-r-2*dx) < .6*(r-l) and abs(y+Y-t-b-2*dy) < .6*(b-t)
                    and score >= .45):
                return max(.65, score)
        return score


def competing_riders(first, second):
    """Two body hypotheses for the same space, with no distinct bike to disambiguate."""
    if first.get("moto") and second.get("moto") and first["moto"] != second["moto"]:
        return False
    l,t,r,b = first["box"]
    x,y,X,Y = second["box"]
    intersection = max(0, min(r,X)-max(l,x)) * max(0, min(b,Y)-max(t,y))
    smaller = min((r-l)*(b-t), (X-x)*(Y-y))
    return (smaller > 0 and intersection >= .75 * smaller
            and abs((l+r-X-x)/2) <= .35 * max(r-l, X-x))


class RiderTracks(Tracks):
    """Track visible occupants through short gaps and abstain on competing body boxes."""
    def update(self, objects, frame):
        supported = [
            person for person in objects
            if person.get("cabezas") or not any(
                other is not person and competing_riders(person, other) for other in objects
            )
        ]
        return super().update(supported, frame)

    def match_score(self, track, obj, gap):
        dx, dy = track.get("velocity", (0., 0.))
        l,t,r,b = track["box"]
        # Limit extrapolation when a detection has been absent for several frames.
        steps = min(8, gap)
        predicted = [l+dx*steps, t+dy*steps, r+dx*steps, b+dy*steps]
        body_score = .7*overlap(predicted, obj["box"]) + .3*overlap(track["box"], obj["box"])
        # A rider leaning into a turn can move faster than the body predictor.
        # Use position relative to a confirmed moving motorcycle as a second cue.
        anchor = obj.get("bike_anchor")
        previous = track.get("bike_anchor")
        if anchor and previous and anchor[0] == previous[0] and gap <= 8:
            bx, by = anchor[1:]
            px, py = previous[1:]
            translated = [l+bx-px, t+by-py, r+bx-px, b+by-py]
            bike_score = overlap(translated, obj["box"])
            if bike_score >= .48:
                return max(body_score, bike_score)
        heads = obj.get("cabezas", [])
        if len(heads) == 1 and "head_frame" in track:
            elapsed = track["last"] + gap - track["head_frame"]
            if elapsed <= 5:
                x,y,X,Y = track["head_box"]
                head_score = overlap([x+dx*elapsed, y+dy*elapsed, X+dx*elapsed, Y+dy*elapsed],
                                     heads[0]["box"])
                # A different head inside an overlapping body should start its own track.
                if head_score == 0:
                    return 0.
                return max(body_score, head_score)
        return body_score

    def observe(self, track, obj, frame):
        track["bike_anchor"] = obj.get("bike_anchor")
        if obj.get("bike_anchor") and obj.get("estado") == "con_casco":
            sightings = track.setdefault("recent_anchored_helmets", [])
            sightings[:] = [f for f in sightings if frame-f <= 12]
            sightings.append(frame)
            if any(h.get("verified_helmet") for h in obj.get("cabezas", [])):
                earlier = track.get("anchored_check", frame)
                if (5 <= frame-earlier <= 12 and earlier > track.get("last_bare", -1)
                        and sum(f >= earlier for f in sightings) >= 6):
                    track["anchored_helmet_frame"] = frame
                if frame-earlier >= 5 or "anchored_check" not in track:
                    track["anchored_check"] = frame
        if "last" in track:
            l,t,r,b = track["box"]
            x,y,X,Y = obj["box"]
            elapsed = frame-track["last"]
            vx,vy = track.get("velocity", (0., 0.))
            track["velocity"] = (.5*vx + .5*(x+X-l-r)/(2*elapsed),
                                 .5*vy + .5*(y+Y-t-b)/(2*elapsed))
        if len(obj.get("cabezas", [])) == 1:
            checks = obj["cabezas"][0].get("partial_checks")
            if checks:
                track.setdefault("partial_checks", set()).update(checks)
            track.update(head_box=obj["cabezas"][0]["box"], head_frame=frame)
        if obj.get("estado") == "sin_casco":
            track.setdefault("first_bare", frame)
            track["last_bare"] = frame
        super().observe(track, obj, frame)

    def state(self, key):
        state = self.detector_state(key)
        track = self.items[key]
        checks = track.get("consensus_frames", [])
        if (state == "indeterminado" and "last_context_bare" not in track
                and len(checks) >= max(6, 2*track["votes"]["sin_casco"], .15*track["hits"])
                and checks[-1]-checks[0] >= 10):
            return "con_casco"
        return state

    def detector_state(self, key):
        state = super().state(key)
        track = self.items[key]
        if (state == "indeterminado" and track.get("anchored_helmet_frame", -1)
                > track.get("last_bare", -1) and track["votes"]["con_casco"] >= 3*track["votes"]["sin_casco"]):
            return "con_casco"
        if state == "sin_casco" and track["last_bare"] - track["first_bare"] < 5:
            return "indeterminado"
        if state != "indeterminado":
            return state
        votes = track["votes"]
        checks = track.get("helmet_checks", [])
        # A short track can end before a third spaced verification is possible.
        # Require two independent checks after any isolated opposite reading.
        if (len(checks) >= 2 and checks[0] > track.get("last_bare", -1)
                and votes["sin_casco"] <= 1 and votes["con_casco"] >= max(6, track["hits"]/3)):
            return "con_casco"
        # Template matches reuse direct sightings. Bound their contribution and
        # require spaced detector verifications before they can resolve a track.
        extra = min(track.get("patch_checks", 0), votes["con_casco"])
        if (extra and len(checks) >= 2 and votes["con_casco"] >= 3
                and votes["con_casco"] >= .8 * sum(votes.values())):
            coverage = (votes["con_casco"] + extra) / track["hits"]
            if coverage >= .5 or (votes["sin_casco"] == 0 and coverage >= 1/3):
                return "con_casco"
        origins = track.get("flow_anchors")
        if origins and max(origins)-min(origins) >= 5 and len(checks) >= 2:
            support = min(track["flow_checks"], votes["con_casco"])
            bare_span = track.get("last_context_bare", 0) - track.get("first_context_bare", 0)
            # Missing boxes do not vote. Two separately anchored motion sequences
            # must be corroborated by fresh head localizations; sustained bare
            # localizations veto this additional path to a helmet confirmation.
            if (support >= 6 and votes["con_casco"] >= 3*votes["sin_casco"]
                    and votes["con_casco"] + support >= 4*votes["sin_casco"] and bare_span < 5):
                return "con_casco"
        return state


RAIN_VIDEO_RUN = "joint-rain-rider-09"
DEFAULT_VIDEO_RUN = "joint-occupant-negatives-11"
PHOTO_RUN = "joint-multiscale-02"


class Stream:
    def __init__(self, source, run, *, motion_check=True, edge_check=True, consensus_check=True, partial_check=True, detector_run=None):
        import cv2
        self.cv = cv2
        self.model = load_model(detector_run or run)
        self.helmet_matcher = None
        if run == RAIN_VIDEO_RUN:
            from rain_helmet_matcher import RainHelmetMatcher
            self.helmet_matcher = RainHelmetMatcher()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError("No se pudo abrir el video o la cámara")
        self.source, self.run = source, detector_run or run
        self.wide_heads = None
        self.temporal_heads = None
        self.motion_heads = None
        self.edge_riders = None
        self.rider_heads = None
        self.portrait_views = False
        self.partial_riders = None
        if not isinstance(source, int) and run != RAIN_VIDEO_RUN:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.portrait_views = run == DEFAULT_VIDEO_RUN and height > width >= 720
            if self.portrait_views or (height > 0 and width >= 1920 and width/height >= 1.45):
                from wide_video import WideVideoHeads
                try:
                    self.wide_heads = WideVideoHeads()
                    if run == DEFAULT_VIDEO_RUN:
                        from temporal_head_search import LiveHelmetSearch
                        self.temporal_heads = LiveHelmetSearch(self.wide_heads)
                        if motion_check:
                            from helmet_flow import HelmetFlow
                            self.motion_heads = HelmetFlow(self.wide_heads, self.cap.get(cv2.CAP_PROP_FPS))
                            if edge_check:
                                from edge_riders import EdgeRiders
                                self.edge_riders = EdgeRiders(self.cap.get(cv2.CAP_PROP_FPS))
                                if consensus_check:
                                    from video_head_consensus import VideoHeadConsensus
                                    self.rider_heads = VideoHeadConsensus(self.model, self.wide_heads, self.edge_riders,
                                                                         self.cap.get(cv2.CAP_PROP_FPS))
                                    if partial_check and not self.portrait_views:
                                        from partial_video_riders import PartialVideoRiders
                                        self.partial_riders = PartialVideoRiders(self.model, self.wide_heads,
                                                                                self.cap.get(cv2.CAP_PROP_FPS))
                except Exception:
                    self.cap.release()
                    raise
        self.bikes = MotorcycleTracks() if self.portrait_views else Tracks()
        self.people = RiderTracks()
        self.frames = 0
        self.seconds = 0.
        self.visible_bikes = self.visible_people = 0

    @property
    def tracking_version(self):
        if self.partial_riders is not None:
            return 9
        if self.rider_heads is not None:
            return 8
        if self.edge_riders is not None:
            return 7
        if self.motion_heads is not None:
            return 6
        return 5 if self.temporal_heads is not None else 4

    def next(self, preview_size=None):
        start = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok:
            if isinstance(self.source, int) or not self.frames:
                raise ValueError("No se pudo leer el fotograma")
            return None
        image = Image.fromarray(self.cv.cvtColor(frame, self.cv.COLOR_BGR2RGB))
        result = analyze_image(self.model, image, draw_boxes=False, video_heads=True)
        self.frames += 1
        assoc = result["association"]
        if self.wide_heads is not None:
            import torch
            from predict import detector_objects
            with torch.inference_mode():
                proposals = result["predictions"]
                detect = lambda view, cutoff: detector_objects(*self.model[:3], view, cutoff)
                if self.portrait_views:
                    from overlapping_views import recover_views, resolved_helmet
                    proposals = recover_views(image, proposals, detect)
                objects = self.wide_heads.refine(image, proposals, detect)
                if self.portrait_views and not summarize(objects)["ocupantes"]:
                    recent = max(self.people.items.values(), key=lambda track: track["last"], default=None)
                    if recent and self.frames-recent["last"] >= 4:
                        box = recent["box"]
                        expanded = recover_views(image, proposals, detect, search_row=(box[1]+box[3])/2)
                        if len(expanded) > len(proposals):
                            checked = self.wide_heads.refine(image, expanded, detect)
                            riders = summarize(checked)["ocupantes"]
                            if riders and all(resolved_helmet(rider) for rider in riders):
                                objects = checked
            assoc = summarize(objects)
        if self.helmet_matcher is not None:
            assoc = summarize(self.helmet_matcher.recover(image, result["predictions"]))
        if self.partial_riders is not None:
            for (body, head), checks in self.partial_riders.find(image, assoc["ocupantes"], self.frames):
                objects = [dict(box=b["box"], class_id=0) for b in assoc["motos"]]
                rider = summarize(objects + [dict(box=body, class_id=1)])["ocupantes"][0]
                rider.update(id=len(assoc["ocupantes"])+1, estado="con_casco", cabezas=[
                    dict(box=head, class_id=2, source="partial_rider", seed_frame=checks[-1],
                         partial_checks=checks)])
                assoc["ocupantes"].append(rider)
        bikes = self.bikes.update(assoc["motos"], self.frames)
        if self.portrait_views:
            for person in assoc["ocupantes"]:
                key = bikes.get(person["moto"])
                if self.bikes.confirmed(key):
                    l,t,r,b = self.bikes.items[key]["box"]
                    person["bike_anchor"] = (key, (l+r)/2, (t+b)/2)
        people = self.people.update(assoc["ocupantes"], self.frames)
        if self.temporal_heads is not None:
            self.temporal_heads.refine(image, assoc["ocupantes"], people, self.people, self.frames)
        if self.motion_heads is not None:
            self.motion_heads.refine(image, assoc["ocupantes"], people, self.people, self.frames)
        if self.edge_riders is not None:
            self.edge_riders.refine(image, assoc["ocupantes"], people, self.people, self.frames)
        if self.rider_heads is not None:
            self.rider_heads.refine(image, assoc["ocupantes"], people, self.people, self.frames)
        assoc["ocupantes"] = [p for p in assoc["ocupantes"] if p["id"] in people]
        for bike in assoc["motos"]:
            bike["id"] = bikes[bike["id"]]
        for person in assoc["ocupantes"]:
            person["id"] = people[person["id"]]
            person["moto"] = bikes.get(person["moto"])
            if not self.bikes.confirmed(person["moto"]):
                person["moto"] = None
            person["estado"] = self.people.state(person["id"])
            if person["estado"] == "indeterminado":
                person["cabezas"] = [h for h in person["cabezas"] if h["class_id"] == 2 and
                                     (h.get("verified_helmet") or h.get("source") in ("temporal_patch_check", "helmet_flow"))]
            elif any(
                (head["class_id"] == 2) != (person["estado"] == "con_casco") for head in person["cabezas"]
            ):
                person["cabezas"] = []
        assoc["motos"] = [b for b in assoc["motos"] if self.bikes.confirmed(b["id"])]
        if self.portrait_views:
            for bike in assoc["motos"]:
                bike["box"] = self.bikes.items[bike["id"]]["box"]
            assoc["motos"] = list({b["id"]: b for b in assoc["motos"]}.values())
        assoc["ocupantes"] = [p for p in assoc["ocupantes"] if self.people.confirmed(p["id"])]
        self.visible_bikes = len(assoc["motos"])
        self.visible_people = len(assoc["ocupantes"])
        if preview_size:
            width, height = image.size
            image.thumbnail(preview_size)
            sx, sy = image.width/width, image.height/height

            def display_box(obj):
                l,t,r,b = obj["box"]
                return dict(obj, box=[l*sx, t*sy, r*sx, b*sy])

            assoc = dict(motos=[display_box(b) for b in assoc["motos"]],
                         ocupantes=[dict(display_box(p), cabezas=[display_box(h) for h in p["cabezas"]])
                                    for p in assoc["ocupantes"]])
        annotate(image, assoc, show_track_state=True)
        self.seconds += time.perf_counter()-start
        return image

    def save(self, root, status):
        self.cap.release()
        if not self.frames:
            return None
        folder = root/"reports"/"videos"/uuid4().hex
        folder.mkdir(parents=True)
        data = dict(rain_helmet_matching=self.helmet_matcher is not None, tracking_version=self.tracking_version, head_crop_consensus=True, run=self.run, source=self.source, status=status, frames=self.frames,
                    timestamp=datetime.now().astimezone().isoformat(),
                    processing_fps=self.frames/self.seconds,
                    counts=dict(motos=self.bikes.count(), ocupantes=self.people.count(), **self.people.state_counts()))
        data["revision_video_panoramico"] = self.wide_heads is not None and not self.portrait_views
        data["revision_video_vertical"] = self.portrait_views
        data["verificacion_temporal_de_cascos"] = self.temporal_heads is not None
        data["trayectorias_ocupantes"] = self.people.evidence()
        tmp = folder/"video.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(folder/"video.json")
        return folder


def video_window(parent, root, run, camera=False):
    source = 0 if camera else filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")])
    if source == "":
        return
    recording = None
    try:
        if not camera:
            from video_recording import Recording, analysis_key, cached_video, play_saved
            directory = root/"reports"/"videos"
            key = analysis_key(root, source, run)
            cached = cached_video(directory, key)
            if cached:
                return play_saved(parent, *cached)
        stream = Stream(source, run)
        if not camera:
            recording = Recording(directory, stream.cap.get(stream.cv.CAP_PROP_FPS))
    except Exception as error:
        messagebox.showerror("Video", str(error), parent=parent)
        return
    win = tk.Toplevel(parent)
    win.title(f"Conteo de motociclistas · {run} · seguimiento v{stream.tracking_version}")
    preview = ttk.Label(win)
    preview.pack()
    note = tk.StringVar()
    ttk.Label(win, textvariable=note).pack()
    pending = None
    closed = False
    fps = stream.cap.get(stream.cv.CAP_PROP_FPS)
    total = stream.cap.get(stream.cv.CAP_PROP_FRAME_COUNT)
    total = int(total) if not camera and math.isfinite(total) and total > 0 else None
    interval = 1/fps if math.isfinite(fps) and fps > 0 and not camera else 0

    def finish(status):
        nonlocal closed
        if closed:
            return
        closed = True
        if pending:
            win.after_cancel(pending)
        try:
            folder = stream.save(root, status)
            if folder:
                if recording and status == "completo" and (not total or stream.frames == total):
                    recording.finish(folder, key, stream.frames)
                c = stream.people.state_counts()
                messagebox.showinfo("Resultado",
                    f"{stream.frames} fotogramas procesados\n"
                    f"Trayectorias: {stream.bikes.count()} de moto y {stream.people.count()} de ocupante\n"
                    f"Con casco: {c['con_casco']} | Sin casco: {c['sin_casco']} | Indeterminados: {c['indeterminado']}\n"
                    f"Guardado en {folder}", parent=win)
        except Exception as error:
            messagebox.showerror("Video", str(error), parent=win)
        finally:
            if recording:
                recording.close()
            win.destroy()

    def tick():
        nonlocal pending
        pending = None
        start = time.perf_counter()
        try:
            image = stream.next(preview_size=(850, 500))
            if image is None:
                finish("completo")
                return
            if recording:
                recording.append(image)
            preview.photo = ImageTk.PhotoImage(image)
            preview.configure(image=preview.photo)
            c = stream.people.state_counts()
            note.set(
                f"Fotograma {stream.frames}{f' de {total}' if total else ''} | "
                f"En pantalla | Motos: {stream.visible_bikes} | Ocupantes: {stream.visible_people}\n"
                f"Trayectorias acumuladas | Motos: {stream.bikes.count()} | Ocupantes: {stream.people.count()}\n"
                f"Con casco: {c['con_casco']} | Sin casco: {c['sin_casco']} | Indeterminados: {c['indeterminado']}"
            )
            pending = win.after(max(1,round(1000*(interval-(time.perf_counter()-start)))),tick)
        except Exception as error:
            messagebox.showerror("Video", str(error), parent=win)
            finish("error")

    ttk.Button(win,text="Detener y guardar",command=lambda:finish("detenido")).pack()
    win.protocol("WM_DELETE_WINDOW", lambda:finish("detenido"))
    pending = win.after(1,tick)
    return win


def query(root, run, question):
    q = question.casefold()
    fields = {"sin casco":"sin_casco","con casco":"con_casco","indeterminad":"indeterminado",
              "ocupante":"ocupantes","motociclista":"ocupantes","motos":"motos","motocicletas":"motos"}
    selected = list(dict.fromkeys(key for word,key in fields.items() if word in q))
    if not selected and not any(word in q for word in ("resumen","estadística","estadistica")):
        return "Puedo consultar motos, ocupantes, con casco, sin casco e indeterminados. Puedes indicar fotos, videos u hoy."
    totals = Counter()
    files = 0
    for kind, pattern in (("foto","fotos/*/foto.json"),("video","videos/*/video.json")):
        if ("video" in q and "foto" not in q and kind=="foto") or ("foto" in q and "video" not in q and kind=="video"):
            continue
        for path in (root/"reports").glob(pattern):
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            selected_runs = {run}
            if kind == "foto" and run in (RAIN_VIDEO_RUN, DEFAULT_VIDEO_RUN):
                selected_runs = {PHOTO_RUN}
            elif kind == "video" and run == DEFAULT_VIDEO_RUN:
                selected_runs.add(PHOTO_RUN)  # Earlier general videos and camera recordings.
            if data.get("run") not in selected_runs:
                continue
            date = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.fromtimestamp(path.stat().st_mtime)
            if "hoy" in q and date.date()!=datetime.now().date():
                continue
            totals.update(data["counts"])
            files+=1
    if not files:
        return "Todavía no hay análisis guardados del modelo para esa consulta."
    values = ", ".join(f"{k.replace('_',' ')}: {totals[k]}" for k in selected or dict.fromkeys(fields.values()))
    return f"{files} análisis guardados · {values}"


def statistics_window(parent,root,run=DEFAULT_VIDEO_RUN):
    win=tk.Toplevel(parent)
    win.title("Asistente de estadísticas")
    ttk.Label(win,text="Ejemplo: ¿Cuántos ocupantes sin casco hubo hoy en videos?").pack(padx=12,pady=6)
    entry=ttk.Entry(win,width=65)
    entry.pack()
    answer=tk.StringVar()
    def ask(*_):
        answer.set(query(root,run,entry.get()))
    entry.bind("<Return>",ask)
    ttk.Button(win,text="Consultar",command=ask).pack()
    ttk.Label(win,textvariable=answer,wraplength=520).pack(padx=12,pady=8)
