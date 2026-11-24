"""Inference using only a checkpoint produced by this project's training runs."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from models import HelmetCNN, build_detector, forbid_external_weights
from association import associate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=float, default=.5)
    a = p.parse_args()
    run = (Path('runs') / a.run).resolve()
    if not run.is_relative_to(Path('runs').resolve()):
        raise RuntimeError('Use a local project run, not external weights.')
    forbid_external_weights()
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config['pretrained'] is not False or config['initialization'] != 'random':
        raise RuntimeError('Checkpoint does not satisfy the from-scratch contract.')
    task = config['task']
    model = HelmetCNN() if task == 'classifier' else build_detector(config.get('num_classes_including_background', 4), config.get('feature_stride',16), config.get('proposal_multiplier',1))
    model.load_state_dict(checkpoint['model'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    with Image.open(a.image) as im:
        image = im.convert('RGB')
    result = {'run': a.run, 'epoch': checkpoint['epoch'], 'task': task, 'image': a.image, 'predictions': []}
    with torch.inference_mode():
        if task == 'classifier':
            crop = ImageOps.pad(image, (96, 96), method=Image.Resampling.BILINEAR, color=(127, 127, 127))
            probs = model(pil_to_tensor(crop).float().div(255).sub(.5).div(.5)[None].to(device)).softmax(1)[0].cpu()
            result['probabilities'] = dict(zip(['con_casco', 'sin_casco'], probs.tolist()))
            result['class'] = ['con_casco', 'sin_casco'][int(probs.argmax())]
        else:
            # Match the deterministic size cap used by the training dataset.
            w, h = image.size
            scale = min(1., 640 / max(w, h))
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
            prediction = model([pil_to_tensor(resized).float().div(255).to(device)])[0]
            draw = ImageDraw.Draw(image)
            names = ['background'] + config.get('classes', ['moto_con_ocupante', 'cabeza_con_casco', 'cabeza_sin_casco'])
            colors = ['white', 'lime', 'cyan', 'red', 'orange']
            for box, cls, score in zip(prediction['boxes'].cpu(), prediction['labels'].cpu(), prediction['scores'].cpu()):
                if float(score) < a.threshold:
                    continue
                coords = (box * torch.tensor([w / nw, h / nh, w / nw, h / nh])).tolist()
                result['predictions'].append({'class_id': int(cls)-1, 'class': names[int(cls)], 'score': float(score), 'box': coords})
                draw.rectangle(coords, outline=colors[int(cls)], width=3)
                draw.text((coords[0], coords[1]), f'{names[int(cls)]} {float(score):.2f}', fill=colors[int(cls)])
            result['association'] = ({'associations': [], 'occupancy_confirmed': False, 'counting_allowed': False,
                                      'limitation': 'Four-class association not implemented; old combined-box heuristic is incompatible.'}
                                     if config.get('separate_classes') else associate(result['predictions'], threshold=a.threshold))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(result, indent=2))
    if task == 'detector':
        image.save(out.with_suffix('.jpg'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
