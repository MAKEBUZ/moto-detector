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
        if "last" in track:
            l,t,r,b = track["box"]
            x,y,X,Y = obj["box"]
            elapsed = frame-track["last"]
            vx,vy = track.get("velocity", (0., 0.))
            track["velocity"] = (.5*vx + .5*(x+X-l-r)/(2*elapsed),
                                 .5*vy + .5*(y+Y-t-b)/(2*elapsed))
        if len(obj.get("cabezas", [])) == 1:
            track.update(head_box=obj["cabezas"][0]["box"], head_frame=frame)
        if obj.get("estado") == "sin_casco":
            track.setdefault("first_bare", frame)
            track["last_bare"] = frame
        super().observe(track, obj, frame)

    def state(self, key):
        state = super().state(key)
        track = self.items[key]
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
        return state


RAIN_VIDEO_RUN = "joint-rain-rider-09"
DEFAULT_VIDEO_RUN = "joint-occupant-negatives-11"
PHOTO_RUN = "joint-multiscale-02"


class Stream:
    def __init__(self, source, run):
        import cv2
        self.cv = cv2
        self.model = load_model(run)
        self.helmet_matcher = None
        if run == RAIN_VIDEO_RUN:
            from rain_helmet_matcher import RainHelmetMatcher
            self.helmet_matcher = RainHelmetMatcher()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError("No se pudo abrir el video o la cámara")
        self.source, self.run = source, run
        self.wide_heads = None
        if not isinstance(source, int) and run != RAIN_VIDEO_RUN:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if height > 0 and width >= 1920 and width/height >= 1.45:
                from wide_video import WideVideoHeads
                try:
                    self.wide_heads = WideVideoHeads()
                except Exception:
                    self.cap.release()
                    raise
        self.bikes, self.people = Tracks(), RiderTracks()
        self.frames = 0
        self.seconds = 0.
        self.visible_bikes = self.visible_people = 0

    def next(self):
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
                objects = self.wide_heads.refine(image, result["predictions"],
                    lambda view, cutoff: detector_objects(*self.model[:3], view, cutoff))
            assoc = summarize(objects)
        if self.helmet_matcher is not None:
            assoc = summarize(self.helmet_matcher.recover(image, result["predictions"]))
        bikes = self.bikes.update(assoc["motos"], self.frames)
        people = self.people.update(assoc["ocupantes"], self.frames)
        assoc["ocupantes"] = [p for p in assoc["ocupantes"] if p["id"] in people]
        for bike in assoc["motos"]:
            bike["id"] = bikes[bike["id"]]
        for person in assoc["ocupantes"]:
            person["id"] = people[person["id"]]
            person["moto"] = bikes.get(person["moto"])
            if not self.bikes.confirmed(person["moto"]):
                person["moto"] = None
            person["estado"] = self.people.state(person["id"])
            if person["estado"] == "indeterminado" or any(
                (head["class_id"] == 2) != (person["estado"] == "con_casco") for head in person["cabezas"]
            ):
                person["cabezas"] = []
        assoc["motos"] = [b for b in assoc["motos"] if self.bikes.confirmed(b["id"])]
        assoc["ocupantes"] = [p for p in assoc["ocupantes"] if self.people.confirmed(p["id"])]
        self.visible_bikes = len(assoc["motos"])
        self.visible_people = len(assoc["ocupantes"])
        annotate(image, assoc, show_track_state=True)
        self.seconds += time.perf_counter()-start
        return image

    def save(self, root, status):
        self.cap.release()
        if not self.frames:
            return None
        folder = root/"reports"/"videos"/uuid4().hex
        folder.mkdir(parents=True)
        data = dict(rain_helmet_matching=self.helmet_matcher is not None, tracking_version=4, head_crop_consensus=True, run=self.run, source=self.source, status=status, frames=self.frames,
                    timestamp=datetime.now().astimezone().isoformat(),
                    processing_fps=self.frames/self.seconds,
                    counts=dict(motos=self.bikes.count(), ocupantes=self.people.count(), **self.people.state_counts()))
        data["revision_video_panoramico"] = self.wide_heads is not None
        data["trayectorias_ocupantes"] = self.people.evidence()
        tmp = folder/"video.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(folder/"video.json")
        return folder


def video_window(parent, root, run, camera=False):
    source = 0 if camera else filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")])
    if source == "":
        return
    try:
        stream = Stream(source, run)
    except Exception as error:
        messagebox.showerror("Video", str(error), parent=parent)
        return
    win = tk.Toplevel(parent)
    win.title(f"Conteo de motociclistas · {run}")
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
                c = stream.people.state_counts()
                messagebox.showinfo("Resultado",
                    f"{stream.frames} fotogramas procesados\n"
                    f"Trayectorias: {stream.bikes.count()} de moto y {stream.people.count()} de ocupante\n"
                    f"Con casco: {c['con_casco']} | Sin casco: {c['sin_casco']} | Indeterminados: {c['indeterminado']}\n"
                    f"Guardado en {folder}", parent=win)
        finally:
            win.destroy()

    def tick():
        nonlocal pending
        pending = None
        start = time.perf_counter()
        try:
            image = stream.next()
            if image is None:
                finish("completo")
                return
            image.thumbnail((850,500))
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
