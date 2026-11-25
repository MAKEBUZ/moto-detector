"""Inference using only a checkpoint produced by this project's training runs."""
import argparse
import hashlib
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
    result = predict(a.run, a.image, a.output, a.threshold)
    print(json.dumps(result), flush=True)


def predict(run_name, image_path, output, threshold=.5, checkpoint_name='best.pt'):
    loaded = load_model(run_name, checkpoint_name)
    with Image.open(image_path) as im:
        image = ImageOps.exif_transpose(im).convert('RGB')
    result = analyze_image(loaded, image, threshold)
    result["image"] = str(image_path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(result, indent=2))
    if result['task'] == 'detector':
        image.save(out.with_suffix('.jpg'))
    return result



def load_model(run_name, checkpoint_name='best.pt'):
    if checkpoint_name not in ('best.pt', 'last.pt'):
        raise ValueError('Checkpoint o umbral invalido')
    run = (Path('runs') / run_name).resolve()
    if not run.is_relative_to(Path('runs').resolve()):
        raise RuntimeError('Use a local project run, not external weights.')
    forbid_external_weights()
    checkpoint = torch.load(run / checkpoint_name, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    random_start = config['pretrained'] is False and config['initialization'] == 'random'
    local_transfer = False
    sources = {'own_localization_transfer': 'edge-localization', 'own_clean_transfer': 'edge-clean'}
    if config['initialization'] in sources and config.get('external_pretrained') is False:
        source = Path('runs') / f"{sources[config['initialization']]}-{int(config['seed'])}" / 'transfer.pt'
        local_transfer = hashlib.sha256(source.read_bytes()).hexdigest() == config.get('transfer_sha256')
    if config["initialization"] == "all_photos_transfer" and config.get("external_pretrained") is False:
        source = Path("runs/clean-heads960-12/best.pt")
        local_transfer = hashlib.sha256(source.read_bytes()).hexdigest() == config.get("transfer_sha256")
    parents = {"joint_project_transfer": "runs/detector-clean-full-01/last.pt",
               "project_consistency": "runs/joint-clean640-01/best.pt",
               "joint_multiscale_transfer": "runs/joint-clean640-01/best.pt"}
    parents["joint_context_transfer"] = "runs/joint-multiscale-02/best.pt"
    parents["occupant_region_transfer"] = "runs/joint-reviewed-08-10/best.pt"
    if config.get("parent_run") in ("joint-occupants-04","joint-multiscale-02"):
        parents["joint_lowrate_transfer"]=f"runs/{config['parent_run']}/best.pt"
    if config["initialization"] in parents and config.get("external_pretrained") is False:
        parent = Path(parents[config["initialization"]])
        local_transfer = hashlib.sha256(parent.read_bytes()).hexdigest() == config.get("parent_sha256")
    if not (random_start or local_transfer):
        raise RuntimeError('Checkpoint does not satisfy the from-scratch contract.')
    task = config['task']
    if config.get('deterministic_gpu'):
        from deterministic_gpu import configure_deterministic_gpu
        configure_deterministic_gpu()
    model = HelmetCNN() if task == 'classifier' else build_detector(config.get('num_classes_including_background', 4), config.get('feature_stride',16), config.get('proposal_multiplier',1), deterministic_roi=config.get('deterministic_gpu',False), small_head_anchors=config.get('small_head_anchors',False), image_size=config.get('image_size',640))
    model.load_state_dict(checkpoint['model'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, config, device, run_name, checkpoint_name, checkpoint['epoch']


def detector_objects(model, config, device, image, threshold):
    objects = []
    # Match the deterministic size cap used by the training dataset.
    w, h = image.size
    scale = min(1., config.get('image_size',640) / max(w, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    prediction = model([pil_to_tensor(resized).float().div(255).to(device)])[0]
    names = ['background'] + config.get('classes', ['moto_con_ocupante', 'cabeza_con_casco', 'cabeza_sin_casco'])
    for box, cls, score in zip(prediction['boxes'].cpu(), prediction['labels'].cpu(), prediction['scores'].cpu()):
        if float(score) < threshold:
            continue
        coords = (box * torch.tensor([w / nw, h / nh, w / nw, h / nh])).tolist()
        objects.append({'class_id': int(cls)-1, 'class': names[int(cls)], 'score': float(score), 'box': coords})
    return objects


def analyze_image(loaded, image, threshold=.5, draw_boxes=True, border_check=True, video_heads=False):
    if not 0 <= threshold <= 1:
        raise ValueError("Umbral fuera de rango")
    model, config, device, run_name, checkpoint_name, epoch = loaded
    task = config['task']
    result = {'run': run_name, 'epoch': epoch, 'task': task, 'checkpoint': checkpoint_name, 'threshold': threshold, 'predictions': []}
    with torch.inference_mode():
        if task == 'classifier':
            crop = ImageOps.pad(image, (96, 96), method=Image.Resampling.BILINEAR, color=(127, 127, 127))
            probs = model(pil_to_tensor(crop).float().div(255).sub(.5).div(.5)[None].to(device)).softmax(1)[0].cpu()
            result['probabilities'] = dict(zip(['con_casco', 'sin_casco'], probs.tolist()))
            result['class'] = ['con_casco', 'sin_casco'][int(probs.argmax())]
        else:
            def detect(view, cutoff):
                return detector_objects(model, config, device, view, cutoff)
            objects = detect(image, threshold)
            if config.get('separate_classes'):
                from photo_results import suppress_nested_bikes
                objects = suppress_nested_bikes(objects)
                from crowded_riders import reconsider, recover_hidden_people
                objects = reconsider(image, objects, detect, threshold)
                objects = recover_hidden_people(image, objects, detect, threshold)
            if config.get('separate_classes') and border_check:
                from border_heads import recover, recover_single_rider
                objects.extend(recover(image, objects, detect, threshold))
                objects.extend(recover_single_rider(image, objects, detect, threshold))
            if video_heads and config.get('separate_classes'):
                from video_heads import add_heads
                objects = add_heads(image, objects, lambda view: detect(view, max(.75, threshold)))
            result['predictions'] = objects
            if draw_boxes and not config.get('separate_classes'):
                draw = ImageDraw.Draw(image)
                colors = ['lime', 'cyan', 'red', 'orange']
                for obj in objects:
                    coords = obj['box']
                    color = colors[obj['class_id']]
                    draw.rectangle(coords, outline=color, width=3)
                    draw.text((coords[0],coords[1]), f"{obj['class']} {obj['score']:.2f}", fill=color)
            if config.get('separate_classes'):
                from photo_results import summarize, annotate
                result['association'] = summarize(result['predictions'])
                result['counts'] = result['association']['counts']
                if draw_boxes:
                    annotate(image, result['association'])
            else:
                result['association'] = associate(result['predictions'], threshold=threshold)
    return result


if __name__ == '__main__':
    main()
