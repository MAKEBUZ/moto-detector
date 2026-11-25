import argparse
import hashlib
import json
import random
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from models import HelmetCNN, build_detector, forbid_external_weights
from data import HelmetDataset, SeparateHelmetDataset, collate_detection
from metrics import classifier_metrics, detection_ap50


def fingerprint(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def evaluate(model, loader, device, task, class_ids=(1, 2, 3)):
    model.eval()
    confusion = torch.zeros(2, 2, dtype=torch.int64)
    predictions, targets = [], []
    for images, truth in loader:
        if task == 'classifier':
            pred = model(images.to(device)).argmax(1).cpu()
            confusion += torch.bincount(truth * 2 + pred, minlength=4).reshape(2, 2)
        else:
            pred = model([x.to(device) for x in images])
            predictions.extend([{k: v.cpu() for k, v in p.items()} for p in pred])
            targets.extend(truth)
    return classifier_metrics(confusion) if task == 'classifier' else detection_ap50(predictions, targets, class_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['classifier', 'detector'], required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit-train', type=int)
    p.add_argument('--limit-val', type=int)
    p.add_argument('--run', required=True)
    p.add_argument('--dataset', default='data/prepared/helmet-baseline-v1')
    p.add_argument('--seed', type=int, default=20261123)
    p.add_argument('--separate-classes', action='store_true', help='Use reviewed four-class dataset')
    p.add_argument('--constant-lr', action='store_true')
    p.add_argument('--photometric', action='store_true')
    p.add_argument('--image-size', type=int, choices=(640, 960), default=640)
    p.add_argument('--feature-stride', type=int, choices=[8,16], default=16)
    p.add_argument('--own-localization-transfer', action='store_true')
    p.add_argument('--all-photos-transfer', action='store_true')
    p.add_argument('--own-clean-transfer', action='store_true')
    p.add_argument('--double-head-objectness', action='store_true')
    p.add_argument('--small-head-anchors', action='store_true', help='Add an 8-pixel RPN anchor scale')
    p.add_argument('--proposal-multiplier', type=int, choices=[1,4], default=1)
    p.add_argument('--head-probe-every', type=int, default=0)
    p.add_argument('--max-updates', type=int, help='Stop after exactly this many optimizer updates; requires constant LR and sufficient epochs')
    p.add_argument('--rpn-object-reserve', action='store_true', help='Reserve one existing positive anchor per GT object; log actual selections')
    p.add_argument('--log-negative-exposure', action='store_true', help='Passively record actual sampler selections near reviewed nonhead regions')
    p.add_argument('--log-reproducibility', action='store_true', help='Record passive step hashes; retain first-step model and gradients')
    p.add_argument('--deterministic-gpu', action='store_true', help='Strict deterministic algorithms and reference RoIAlign; fail on unsupported operations')
    p.add_argument('--reserve-reviewed-negatives', action='store_true')
    a = p.parse_args()
    if a.reserve_reviewed_negatives and not (a.log_negative_exposure and a.deterministic_gpu):
        p.error('Reviewed negative reservation requires exposure logging and deterministic GPU')
    if a.deterministic_gpu:
        from deterministic_gpu import configure_deterministic_gpu
        configure_deterministic_gpu()
        if not torch.cuda.is_available():p.error('Deterministic GPU experiment requires CUDA')
    if a.log_negative_exposure and (not a.separate_classes or a.rpn_object_reserve):
        p.error('Exposure audit requires four classes and unchanged original sampler')
    if a.log_reproducibility and (not a.separate_classes or a.rpn_object_reserve):
        p.error('Reproducibility audit requires original four-class sampler')
    if a.photometric and (not a.separate_classes or a.workers != 0):
        p.error('Photometric controlled experiment requires separate classes and workers=0')
    if a.head_probe_every < 0 or (a.head_probe_every and not a.separate_classes):
        p.error('Head monitoring requires separate classes and a nonnegative interval')
    if a.epochs < 1:
        p.error('epochs must be positive')
    if a.rpn_object_reserve and not a.separate_classes:
        p.error('Object reservation requires the reviewed four-class detector')
    if a.max_updates is not None and (a.max_updates < 1 or not a.constant_lr):
        p.error('max-updates requires a positive limit and constant-lr')
    out = Path('runs') / a.run
    if out.exists():
        raise RuntimeError('Run already exists. Use a new run name; no implicit resume.')
    out.mkdir(parents=True)
    if a.separate_classes:
        if a.task != 'detector': p.error('Separate classes require detector task')
        metadata = json.loads((Path(a.dataset) / 'dataset.json').read_text())
        classes = metadata['classes']
        assert classes == ['motocicleta', 'persona_ocupante', 'cabeza_con_casco', 'cabeza_sin_casco']
        lock_path = Path(a.dataset) / 'dataset.lock.json'
        locked = {str(Path(a.dataset)/name): value for name, value in json.loads(lock_path.read_text()).items()}
    else:
        classes = ['moto_con_ocupante', 'cabeza_con_casco', 'cabeza_sin_casco']
        contract = json.loads((Path(a.dataset) / 'training-contract.json').read_text())
        assert contract['pretrained'] is False and contract['initialization'] == 'random'
        lock_path = Path(a.dataset) / 'dataset-lock.json'
        locked = json.loads(lock_path.read_text())['inputsAndOutputs']
    for name, expected in locked.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Dataset lock mismatch: {name}')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    forbid_external_weights()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HelmetCNN() if a.task == 'classifier' else build_detector(num_classes=len(classes)+1, feature_stride=a.feature_stride, proposal_multiplier=a.proposal_multiplier, deterministic_roi=a.deterministic_gpu, small_head_anchors=a.small_head_anchors, image_size=a.image_size)
    transfer_digest = None
    if a.all_photos_transfer and not a.own_clean_transfer:
        p.error('All-photo transfer requires own-clean-transfer')
    if a.own_clean_transfer:
        if a.own_localization_transfer or not (a.separate_classes and a.deterministic_gpu and a.feature_stride == 8 and a.proposal_multiplier == 4):
            p.error("Clean transfer requires the reviewed stride-8 deterministic detector")
        source = Path(f"runs/edge-clean-{a.seed}/transfer.pt")
        if a.all_photos_transfer:
            source = Path("runs/clean-heads960-12/best.pt")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        assert payload["initialization"] == "random" and payload["external_weights"] is False
        if a.all_photos_transfer:
            assert payload["image_size"] == a.image_size and payload["images"] >= 1583
        else:
            assert payload["epoch"] == 4 and payload["images"] == 1583
        assert payload["classes"] == ["BikeWithRider", "NoHelmet", "Helmet"]
        state = model.state_dict()
        for key, value in payload["model"].items():
            if not key.startswith("roi_heads.box_predictor."):
                state[key] = value
                continue
            width = 4 if ".bbox_pred." in key else 1
            # Only background, helmet and no-helmet have corresponding output rows.
            for old, new in ((0, 0), (3, 3), (2, 4)):
                state[key][new*width:(new+1)*width] = value[old*width:(old+1)*width]
        model.load_state_dict(state)
        transfer_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if a.own_localization_transfer:
        if not (a.separate_classes and a.deterministic_gpu and a.feature_stride==8 and a.proposal_multiplier==4):
            p.error("Local transfer requires the four-class deterministic stride-8 detector")
        source = Path(f"runs/edge-localization-{a.seed}/transfer.pt")
        payload = torch.load(source,map_location="cpu",weights_only=True)
        assert payload["initialization"]=="random" and payload["external_weights"] is False and payload["images"]==1583
        transferred = payload["transfer"]
        state = model.state_dict()
        assert set(transferred)=={k for k in state if k.startswith(("backbone.","rpn."))}
        model.load_state_dict({**state,**transferred})
        transfer_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    initial = fingerprint(model)
    model.to(device)
    head_stats = None
    if a.double_head_objectness:
        if not a.separate_classes or a.rpn_object_reserve or a.log_negative_exposure or a.log_reproducibility:
            p.error("Head weighting requires the four-class detector and original sampler")
        from rpn_head_weight import install_head_weight
        head_stats = install_head_weight(model.rpn)
    reservation = None
    if a.rpn_object_reserve:
        from rpn_object_sampler import install_object_sampler
        reservation = install_object_sampler(model, a.seed + 1000000)
        assert fingerprint(model) == initial
    train = SeparateHelmetDataset(a.dataset, 'train', a.limit_train, photometric=a.photometric, photo_seed=a.seed) if a.separate_classes else HelmetDataset(a.dataset, a.task, 'train', a.limit_train)
    exposure = None
    if a.log_negative_exposure:
        from negative_exposure import ExposureObserver
        train.exposure_regions = True
        exposure = ExposureObserver(model, reserve=a.reserve_reviewed_negatives)
        assert fingerprint(model) == initial
    train.image_size = a.image_size
    val = SeparateHelmetDataset(a.dataset, 'val', a.limit_val, augment=False) if a.separate_classes else HelmetDataset(a.dataset, a.task, 'val', a.limit_val, augment=False)
    val.image_size = a.image_size
    sampler_trace=[]
    if a.log_reproducibility:
        from reproducibility_trace import SamplerTrace, tensors_hash, rng_hashes
        model.rpn.fg_bg_sampler=SamplerTrace(model.rpn.fg_bg_sampler,sampler_trace,'rpn')
        model.roi_heads.fg_bg_sampler=SamplerTrace(model.roi_heads.fg_bg_sampler,sampler_trace,'roi')
    batch = a.batch_size or (64 if a.task == 'classifier' else 2)
    kwargs = {'num_workers': a.workers, 'pin_memory': device.type == 'cuda',
              'persistent_workers': a.workers > 0}
    if a.task == 'detector':
        kwargs['collate_fn'] = collate_detection
    loader = DataLoader(train, batch_size=batch, shuffle=True, **kwargs)
    if a.max_updates is not None and a.max_updates > len(loader) * a.epochs:
        raise ValueError('Epoch budget cannot reach requested updates')
    validation = DataLoader(val, batch_size=batch, shuffle=False, **kwargs)
    lr = .001 if a.task == 'classifier' else .0003
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    if a.own_localization_transfer or a.own_clean_transfer:
        groups = [{"params":[v for k,v in model.named_parameters() if k.startswith(("backbone.","rpn."))],"lr":lr/3},
                  {"params":model.roi_heads.parameters(),"lr":lr}]
        optimizer = torch.optim.AdamW(groups,lr=lr,weight_decay=.0001)
    scheduler = None if a.constant_lr else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    if a.head_probe_every:
        from overfit_heads import probe
        diagnostic_train = SeparateHelmetDataset(a.dataset, 'train', augment=False)
        diagnostic_train.image_size = a.image_size
        probe_samples = {'train': [diagnostic_train[i] for i in range(len(diagnostic_train))],
                         'val': [val[i] for i in range(len(val))]}
        (out/'head-probes').mkdir()
    if a.task == 'classifier':
        counts = torch.bincount(torch.tensor([r['classId'] for r in train.rows]), minlength=2)
        assert (counts > 0).all()
        criterion = torch.nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts.float())).to(device))
    config = {**vars(a), 'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__), 'initialization': 'random',
              'pretrained': False, 'initial_state_sha256': initial, 'parameters': sum(p.numel() for p in model.parameters()),
              'train_examples': len(train), 'validation_examples': len(val), 'test_used': False,
              'learning_rate': lr, 'batch_size': batch, 'classes': classes, 'num_classes_including_background': len(classes)+1,
              'pilot': bool(a.separate_classes or a.limit_train or a.limit_val)}
    if transfer_digest:
        config.update(initialization="own_localization_transfer", pretrained=True, external_pretrained=False,
                      transfer_sha256=transfer_digest, transferred_learning_rate=lr/3)
        if a.own_clean_transfer:
            config["initialization"] = "all_photos_transfer" if a.all_photos_transfer else "own_clean_transfer"
    config['source_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in sorted(Path('training').glob('*.py'))}
    config['dataset_lock_sha256'] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    config['gpu_reproducibility_settings'] = {'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_version':torch.backends.cudnn.version(),'cuda_version':torch.version.cuda,'threads':torch.get_num_threads()}
    config['gpu_reproducibility_settings'].update(deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        roi_align_implementation='torchvision_reference_eager' if a.deterministic_gpu else 'torchvision_default')
    config['photometric_policy'] = ({'brightness':[.85,1.15],'contrast':[.85,1.15],'blur_probability':.25,'blur_radius_pixels':[.2,.8],
                                     'stage':f'after {a.image_size}px cap, before model transform','rng':'independent Python Random seeded with run seed'} if a.photometric else None)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    with (out / 'requirements.lock.txt').open('w') as requirements:
        subprocess.run([str(Path('.venv/Scripts/python.exe').resolve()), '-m', 'pip', 'freeze'], stdout=requirements, check=True)
    print(json.dumps(config), flush=True)
    history, best, best_epoch = [], -1., 0
    selection_log = (out/'rpn-selections.jsonl').open('x', encoding='utf-8') if reservation else None
    exposure_log = (out/'negative-exposure.jsonl').open('x', encoding='utf-8') if exposure else None
    trace_log = (out/'reproducibility.jsonl').open('x', encoding='utf-8') if a.log_reproducibility else None
    updates = presentations = 0
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train(); total, seen = 0., 0; component_totals = {}
        for images, truth in loader:
            optimizer.zero_grad(set_to_none=True)
            if trace_log:
                sampler_trace.clear()
                trace={'update':updates+1,'epoch':epoch,'image_indices':[int(t['image_id']) for t in truth],
                       'inputs_sha256':tensors_hash(images),
                       'targets_sha256':tensors_hash([t[k] for t in truth for k in sorted(t)]),
                       'rng_before':rng_hashes()}
            if a.task == 'classifier':
                loss = criterion(model(images.to(device)), truth.to(device)); size = len(truth)
            else:
                losses = model([x.to(device) for x in images], [{k: v.to(device) for k, v in t.items()} for t in truth])
                loss = sum(losses.values()); size = len(images)
                for name, value in losses.items():
                    component_totals[name] = component_totals.get(name, 0.) + float(value.detach()) * size
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; stopping instead of writing a misleading model.')
            if trace_log:
                trace.update(losses={k:float(v.detach()) for k,v in losses.items()},samplers=list(sampler_trace),rng_after_forward=rng_hashes())
            loss.backward()
            if trace_log and (updates<10 or (updates+1)%25==0):
                trace['gradient_sha256']=tensors_hash([p.grad for p in model.parameters()])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            if trace_log:
                trace['rng_after_step']=rng_hashes()
                if updates<10 or (updates+1)%25==0:trace['model_sha256']=fingerprint(model)
                if updates==0:
                    torch.save({'model':model.state_dict(),'clipped_gradients':{n:p.grad for n,p in model.named_parameters()}},out/'first-step.pt')
                trace_log.write(json.dumps(trace)+'\n')
            total += float(loss.detach()) * size; seen += size
            updates += 1; presentations += size
            if exposure:
                for entry in exposure.records:
                    row = train.rows[entry['image_index']]
                    entry.update(update=updates, epoch=epoch, image_id=row['image_id'], region_id=row['reviewed_nonhead_regions'][entry['region_index']]['id'])
                    exposure_log.write(json.dumps(entry)+'\n')
                exposure.records.clear()
            if reservation:
                assert len(reservation.records) == size
                for entry in reservation.records:
                    row = train.rows[entry['image_index']]
                    entry.update({'update': updates, 'epoch': epoch, 'image_id': row['image_id']})
                    for obj in entry['objects']:
                        obj['entity_id'] = row['annotations'][obj['gt_index']]['entity_id']
                    selection_log.write(json.dumps(entry)+'\n')
                reservation.records.clear()
            if a.max_updates is not None and updates == a.max_updates:
                break
        reached_limit = a.max_updates is not None and updates == a.max_updates
        metrics = evaluate(model, validation, device, a.task, tuple(range(1, len(classes)+1)))
        score = metrics['macro_f1' if a.task == 'classifier' else 'map50']
        record = {'epoch': epoch, 'optimizer_updates': updates, 'image_presentations': presentations, 'train_loss': total / seen, 'train_loss_components': {k:v/seen for k,v in component_totals.items()},
                  'learning_rate': optimizer.param_groups[0]['lr'], 'validation': metrics, 'elapsed_seconds': round(time.time() - start, 2)}
        if a.head_probe_every and (epoch == 1 or epoch % a.head_probe_every == 0 or epoch == a.epochs or reached_limit):
            # Preserve RNG state so monitoring does not alter training sampling.
            with torch.random.fork_rng(devices=[0] if device.type == 'cuda' else []):
                diagnostics = {split: probe(model, samples, device) for split, samples in probe_samples.items()}
            (out/'head-probes'/f'epoch-{epoch:03d}.json').write_text(json.dumps(diagnostics, indent=2))
            record['head_monitor'] = {split: {k:v for k,v in report.items() if k not in ('heads','head_ap50')} for split,report in diagnostics.items()}
        history.append(record); print(json.dumps(record), flush=True)
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                      'optimizer_updates': updates, 'image_presentations': presentations,
                      'config': config, 'validation': metrics, 'initial_state_sha256': initial}
        torch.save(checkpoint, out / 'last.pt')
        if score > best:
            best, best_epoch = score, epoch; torch.save(checkpoint, out / 'best.pt')
        if scheduler is not None: scheduler.step()
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        if selection_log: selection_log.flush()
        if exposure_log: exposure_log.flush()
        if trace_log: trace_log.flush()
        if reached_limit: break
    if a.max_updates is not None: assert updates == a.max_updates
    if head_stats is not None:
        (out / "head-weight-stats.json").write_text(json.dumps(head_stats, indent=2))
    if selection_log: selection_log.close()
    if exposure_log: exposure_log.close()
    if trace_log: trace_log.close()
    result = {'task': a.task, 'epochs': epoch, 'optimizer_updates': updates, 'image_presentations': presentations, 'final_validation': metrics, 'best_epoch': best_epoch, 'best_validation_score': best,
              'selection_metric': 'macro_f1' if a.task == 'classifier' else 'map50',
              'initial_state_sha256': initial, 'final_state_sha256': fingerprint(model),
              'test_used': False, 'pilot': config['pilot'], 'seconds': round(time.time() - start, 2)}
    assert result['initial_state_sha256'] != result['final_state_sha256'], 'Weights did not change'
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
