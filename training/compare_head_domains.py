"""Build a descriptive gallery; matching is by class/size, not semantic similarity."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def main():
    root=Path('data/prepared/helmet-separate-v2-expanded-01')
    report=Path('reports/training/separate-v2-long-01')
    diagnostic=json.loads((report/'best-head-diagnostics/diagnosis.json').read_text())
    out=report/'head-comparison';out.mkdir(parents=True,exist_ok=True)
    manifests={s:list(map(json.loads,(root/f'{s}-manifest.jsonl').read_text().splitlines())) for s in ('train','val')}
    records=[]
    for r in diagnostic['records']:
        scene=next(s for s in manifests[r['split']] if s['image_id']==r['image_id'])
        ann=next(a for a in scene['annotations'] if a['entity_id']==r['entity_id'])
        records.append({**r,'box':ann['box'],'image':scene['image']})
    panels=[];comparisons=[]
    for v in (r for r in records if r['split']=='val'):
        candidates=[r for r in records if r['split']=='train' and r['true_label']==v['true_label']]
        nearest=sorted(candidates,key=lambda r:abs(math.log(r['min_side_model_pixels']/v['min_side_model_pixels'])))[:2]
        card=Image.new('RGB',(960,300),'white');draw=ImageDraw.Draw(card)
        for j,r in enumerate([v]+nearest):
            with Image.open(root/r['image']) as im:
                crop=im.convert('RGB').crop(tuple(r['box']))
                thumb=ImageOps.contain(crop,(280,210))
                card.paste(thumb,(j*320+(320-thumb.width)//2,75))
            oracle=max(range(5),key=lambda k:r['oracle_box_class_probabilities'][k])
            label='HELMET' if r['true_label']==3 else 'NO HELMET'
            draw.text((j*320+8,8),f"{r['split']} {r['image_id']} {r['entity_id']}\n{label}; size={r['min_side_model_pixels']:.1f}px\nRPN IoU={r['best_rpn_iou']:.2f}; oracle={oracle}",fill='black')
        panels.append(card);comparisons.append({'validation':v,'train_size_neighbors':nearest})
    for k in range(0,len(panels),3):
        sheet=Image.new('RGB',(960,300*len(panels[k:k+3])),'#dddddd')
        for j,panel in enumerate(panels[k:k+3]):sheet.paste(panel,(0,j*300))
        sheet.save(out/f'comparison-{k//3+1}.jpg')
    contexts=Image.new('RGB',(1500,600),'white')
    for i,s in enumerate(manifests['val']):
        with Image.open(root/s['image']) as im:
            im=im.convert('RGB');scale=min(300/im.width,550/im.height)
            im=im.resize((round(im.width*scale),round(im.height*scale)))
        draw=ImageDraw.Draw(im)
        for a in s['annotations']:
            if a['classId'] in (2,3):
                b=[v*scale for v in a['box']];draw.rectangle(b,outline='lime',width=2);draw.text((b[0],b[1]),a['entity_id'],fill='red')
        contexts.paste(im,(i*300,40));ImageDraw.Draw(contexts).text((i*300+4,10),s['image_id'],fill='black')
    contexts.save(out/'validation-context.jpg')
    summary={}
    for split in ('train','val'):
        rr=[r for r in records if r['split']==split];sizes=sorted(r['min_side_model_pixels'] for r in rr)
        summary[split]={'heads':len(rr),'size_min':min(sizes),'size_median':sizes[len(sizes)//2],'size_max':max(sizes),
                        'under16':sum(x<16 for x in sizes),'under32':sum(x<32 for x in sizes),
                        'rpn_iou50':sum(r['best_rpn_iou']>=.5 for r in rr),
                        'oracle_correct':sum(max(range(5),key=lambda k:r['oracle_box_class_probabilities'][k])==r['true_label'] for r in rr)}
    (out/'comparison.json').write_text(json.dumps({'checkpoint':diagnostic['checkpoint_sha256'],'epoch':diagnostic['epoch'],
        'pairing':'same class and nearest log model-pixel size; not visual similarity','summary':summary,'cases':comparisons},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
