"""Local desktop photo analysis using the project's deterministic control."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from predict import predict
from traffic_video import DEFAULT_VIDEO_RUN, PHOTO_RUN, statistics_window, video_window

ROOT = Path(__file__).resolve().parents[1]


def analyze(path):
    directory = ROOT / 'reports' / 'fotos'
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as temporary:
        folder = Path(temporary) / 'resultado'
        folder.mkdir()
        result = predict(PHOTO_RUN, path, folder/'foto', .5, 'best.pt')
        destination = directory / uuid4().hex
        folder.rename(destination)
    return result, destination


def main():
    os.chdir(ROOT)
    window = tk.Tk()
    window.title('Detección de casco en motociclistas')
    preview = ttk.Label(window)
    status = tk.StringVar(value='Selecciona una foto para analizar sus ocupantes.')
    def choose():
        path = filedialog.askopenfilename(filetypes=[('Fotografías','*.jpg *.jpeg *.png *.bmp *.webp'),('Todos','*.*')])
        if not path:
            return
        status.set('Analizando la foto…')
        window.update_idletasks()
        try:
            result, folder = analyze(path)
            with Image.open(folder/'foto.jpg') as im:
                im.thumbnail((850,500))
                preview.photo = ImageTk.PhotoImage(im)
            preview.configure(image=preview.photo)
            c = result['counts']
            lines = [f"Motos: {c['motos']}   Ocupantes: {c['ocupantes']}   Con casco: {c['con_casco']}   Sin casco: {c['sin_casco']}   Indeterminados: {c['indeterminado']}"]
            for rider in result['association']['ocupantes']:
                moto = rider['moto'] or 'sin asignar'
                detalle = " (confirmado en dos vistas)" if any(h.get("source") in ("padded_consensus","upper_body_consensus") for h in rider["cabezas"]) else ""
                lines.append(f"Persona {rider['id']} · Moto {moto} · {rider['estado'].replace('_',' ')}{detalle}")
            lines.append(f'Guardado en {folder}')
            status.set('\n'.join(lines))
        except Exception as error:
            status.set('No se pudo analizar la foto.')
            messagebox.showerror('Error de análisis', str(error))
    ttk.Button(window,text='Cargar foto',command=choose).pack(pady=8)
    ttk.Button(window,text="Cargar video",command=lambda: video_window(window, ROOT, DEFAULT_VIDEO_RUN)).pack()
    ttk.Button(window,text="Usar cámara",command=lambda: video_window(window, ROOT, PHOTO_RUN, camera=True)).pack()
    ttk.Button(window,text="Consultar estadísticas",command=lambda: statistics_window(window, ROOT, DEFAULT_VIDEO_RUN)).pack()
    preview.pack()
    ttk.Label(window,textvariable=status,justify='left',wraplength=800).pack(padx=12,pady=8)
    window.mainloop()


if __name__=='__main__':
    main()
