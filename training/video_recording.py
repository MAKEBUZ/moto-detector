"""Persist completed annotated previews and play them on the source time line."""
import hashlib
import json
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageTk


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def analysis_key(root, source, run):
    files = sorted({root / "runs" / name / "best.pt" for name in (
        run, "joint-multiscale-02", "joint-clean640-01", "joint-reviewed-08-10",
        "rider-crop-heads-01", "crop-specialist-01")})
    modules = ("traffic_video", "predict", "models", "association", "photo_results",
               "crowded_riders", "border_heads", "video_heads", "wide_video",
               "rider_crop_heads", "temporal_head_search", "helmet_flow", "edge_riders",
               "video_head_consensus", "partial_video_riders", "overlapping_views",
               "rain_helmet_matcher", "video_recording")
    files.extend(root / "training" / (name + ".py") for name in modules)
    parts = [(p.relative_to(root).as_posix(), digest(p) if p.exists() else None) for p in files]
    return hashlib.sha256(json.dumps([run, digest(source), parts]).encode()).hexdigest()


def cached_video(directory, key):
    for report in directory.glob("*/video.json"):
        try:
            data = json.loads(report.read_text(encoding="utf-8-sig"))
            clip = report.parent / "anotado.avi"
            if (data.get("status") == "completo" and data.get("analysis_key") == key
                    and clip.is_file() and data.get("annotated_sha256") == digest(clip)):
                return clip, data
        except (OSError, ValueError):
            continue
    return None


class Recording:
    def __init__(self, directory, fps):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (uuid4().hex + ".partial.avi")
        self.writer = None
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30

    def append(self, image):
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        bgr = cv2.copyMakeBorder(bgr, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE)
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"),
                                          self.fps, (bgr.shape[1], bgr.shape[0]))
            if not self.writer.isOpened():
                raise OSError("No se pudo crear el video anotado.")
        self.writer.write(bgr)

    def finish(self, folder, key, frames):
        self.writer.release()
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
                raise OSError("El video anotado está incompleto.")
        finally:
            cap.release()
        clip = self.path.replace(folder / "anotado.avi")
        report = folder / "video.json"
        data = json.loads(report.read_text(encoding="utf-8"))
        data.update(analysis_key=key, annotated_video=clip.name, annotated_sha256=digest(clip))
        tmp = report.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(report)

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.path.unlink(missing_ok=True)


def play_saved(parent, clip, data):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        cap.release()
        raise OSError("No se pudo abrir la reproducción guardada.")
    win = tk.Toplevel(parent)
    win.title("Reproducción guardada · " + data["run"])
    preview = ttk.Label(win)
    preview.pack()
    c = data["counts"]
    ttk.Label(win, text=(f"Resultado guardado | Motos: {c['motos']} | Ocupantes: {c['ocupantes']} | "
                         f"Con casco: {c['con_casco']} | Sin casco: {c['sin_casco']} | "
                         f"Indeterminados: {c['indeterminado']}")).pack()
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = time.perf_counter()
    position = 0
    pending = None

    def close():
        if pending is not None:
            win.after_cancel(pending)
        cap.release()
        win.destroy()

    def tick():
        nonlocal pending, position
        pending = None
        target = min(data["frames"]-1, int((time.perf_counter()-start)*fps))
        while position < target and cap.grab():
            position += 1
        ok, frame = cap.read()
        if not ok:
            if position < data["frames"]:
                messagebox.showerror("Video", "Error al leer la reproducción guardada.", parent=win)
            close()
            return
        position += 1
        preview.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        preview.configure(image=preview.photo)
        remaining = start + position/fps - time.perf_counter()
        pending = win.after(max(1, round(remaining*1000)), tick)

    win.protocol("WM_DELETE_WINDOW", close)
    pending = win.after(1, tick)
    return win

