from pathlib import Path
import json
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from deterministic_gpu import configure_deterministic_gpu
from data import SeparateHelmetDataset
from models import build_detector
from train import fingerprint
from collections import OrderedDict

@torch.inference_mode()
def main():
    configure_deterministic_gpu()
    base = Path('reports/training/reviewed-negative-reserve-01')
    protocol = json.loads((base/'protocol.json').read_text())
    rows = []
    for kind in ('control','variant'):
        checkpoint = torch.load(Path('runs')/protocol[kind]/'last.pt',map_location='cpu',weights_only=True)
        config = checkpoint['config']
        model = build_detector(5,config['feature_stride'],config['proposal_multiplier'],deterministic_roi=True).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        initial = fingerprint(model)
        for split in ('train','val'):
            dataset = SeparateHelmetDataset(Path(config['dataset']),split,augment=False)
            for index,(image,target) in enumerate(dataset):
                for j in torch.where(target['labels']>=3)[0].tolist():
                    x,y,X,Y = target['boxes'][j].tolist()
                    side = max(X-x,Y-y)
                    for factor in (1.,1.5,2.):
                        # Same center, square window. Fill outside image with reflection-free edge replication.
                        for pixels in (64,128):
                            axis = ((torch.arange(pixels,device='cuda')+.5)/pixels-.5)*factor*side
                            gy,gx = torch.meshgrid(axis+(y+Y)/2,axis+(x+X)/2,indexing='ij')
                            grid = torch.stack((2*gx/image.shape[2]-1,2*gy/image.shape[1]-1),-1)
                            crop = torch.nn.functional.grid_sample(image.cuda()[None],grid[None],align_corners=False,padding_mode='border')[0]
                            features = OrderedDict([('0',model.backbone(model.transform.normalize(crop)[None]))])
                            w,h = (X-x)*pixels/(factor*side),(Y-y)*pixels/(factor*side)
                            box = crop.new_tensor([[(pixels-w)/2,(pixels-h)/2,(pixels+w)/2,(pixels+h)/2]])
                            roi = model.roi_heads
                            pooled = roi.box_roi_pool(features,[box],[(pixels,pixels)])
                            logits,_ = roi.box_predictor(roi.box_head(pooled))
                            scores = logits[0].softmax(0)
                            rows.append(dict(kind=kind,split=split,image=dataset.rows[index]['image_id'],index=j,label=int(target['labels'][j]),factor=factor,pixels=pixels,scores=scores.tolist(),binary=int(scores[3:].argmax())+3,winner=int(scores.argmax())))
        assert initial==fingerprint(model)
    out=base/'crop-context'
    out.mkdir(exist_ok=True)
    (out/'scores.json').write_text(json.dumps(rows))
    lines=['# Tamaño y contexto de recortes con pesos congelados','','Cajas reales, ventanas cuadradas centradas, borde replicado y dos resoluciones de entrada al backbone. El clasificador recibe la caja de cabeza en coordenadas del recorte. Cada columna binaria indica aciertos condicionados a una cabeza conocida. Fondo cuenta las veces que gana entre las cinco clases.','','| Modelo | Partición | Ventana × lado mayor | Píxeles | Con casco | Sin casco | Fondo |','|---|---|---:|---:|---:|---:|---:|']
    for kind in ('control','variant'):
        for split in ('train','val'):
            for factor in (1.,1.5,2.):
                for pixels in (64,128):
                    group=[r for r in rows if (r['kind'],r['split'],r['factor'],r['pixels'])==(kind,split,factor,pixels)]
                    hits=[f"{sum(r['binary']==k and r['label']==k for r in group)}/{sum(r['label']==k for r in group)}" for k in (3,4)]
                    lines.append(f"| {kind} | {split} | {factor} | {pixels} | {hits[0]} | {hits[1]} | {sum(r['winner']==0 for r in group)}/{len(group)} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__':
    main()
