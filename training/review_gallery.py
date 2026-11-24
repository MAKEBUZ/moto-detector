import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

out=Path('reports/error-review')
queue=json.loads((out/'association-review-queue.json').read_text())
canvas=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(queue):
    x=(i%3)*400;y=(i//3)*300
    with Image.open(out/r['annotated_preview']) as im:tile=ImageOps.contain(im,(390,270))
    canvas.paste(tile,(x,y+25));draw.text((x+3,y+3),r['image_id'],fill='black')
canvas.save(out/'association-gallery.jpg')
errors=json.loads((out/'classifier-errors.json').read_text())[:12]
rows={r['id']:r for r in (json.loads(l) for l in Path('data/prepared/helmet-baseline-v1/classifier-manifest.jsonl').read_text().splitlines())}
canvas=Image.new('RGB',(1000,750),'white');draw=ImageDraw.Draw(canvas)
for i,error in enumerate(errors):
    r=rows[error['id']];x=(i%4)*250;y=(i//4)*250
    path=Path('data/prepared/helmet-baseline-v1/classifier/val')/r['className']/(r['id']+'.jpg')
    with Image.open(path) as im:tile=ImageOps.contain(im,(240,205))
    canvas.paste(tile,(x,y+40));draw.text((x+3,y+3),f"{r['id'][:10]} real={error['true_class']} pred={error['predicted_class']}\nscore={error['confidence']:.3f} (0=con,1=sin)",fill='black')
canvas.save(out/'classifier-errors-gallery.jpg')
