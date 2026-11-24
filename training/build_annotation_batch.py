"""Build an assistant-reviewed draft, never training-ready ground truth."""
import base64
import hashlib
import io
import json
from pathlib import Path
from PIL import Image

ROOT=Path('data/annotations/relations-pilot-v2')
ROOT.mkdir(parents=True,exist_ok=True)
originals={Path(r['image']).name:r for r in (json.loads(l) for l in Path('reports/edgevision-manifest.jsonl').read_text().splitlines())}
decisions={r['image']:r for r in (json.loads(l) for l in Path('reports/preparation/image-decisions.jsonl').read_text().splitlines())}
specs={
 'Image_00146.jpg':{'vehicles':[1],'heads':[(0,'sin_casco',0)],'note':'Las clases originales están intercambiadas: la caja 0 delimita cabeza sin casco y la 1 moto con ocupante. Conservar geometría como propuesta, no aprobar automáticamente.'},
 'Image_01898.jpg':{'vehicles':[0],'heads':[(1,'sin_casco',0)],'note':'Se observa cabello en la cabeza originalmente etiquetada Helmet. Propuesta: sin casco.'},
 'Image_01575.jpg':{'vehicles':[0,1,5],'heads':[(3,'sin_casco',0),(2,'con_casco',1),(4,'con_casco',2)],'note':'La cabeza izquierda lleva pañuelo: conservar sin casco (anotación 3). La cabeza del motociclista más lejano parece llevar casco: revisar cambio en anotación 4. Tres motos distintas.'},
 'Image_01720.jpg':{'vehicles':[0],'heads':[(1,'sin_casco',0)],'note':'El ocupante sentado pertenece a la moto. La persona de pie detrás NO debe asignarse a ella.'},
 'Image_00135.jpg':{'vehicles':[0,1],'heads':[(2,'sin_casco',0),(3,'sin_casco',1)],'note':'Vínculos propuestos para primer plano y moto lejana. Revisar oclusiones y visibilidad.'},
 'Image_02097.jpg':{'vehicles':[0,3],'heads':[(1,'sin_casco',0),(2,'sin_casco',1),(4,'con_casco',None)],'note':'Dos motos próximas. La cabeza izquierda se propone para la moto izquierda. La cabeza parcialmente ocluida con casco queda con vínculo incierto.'},
 'Image_02354.jpg':{'vehicles':[0],'heads':[(1,'con_casco',0)],'note':'Una cabeza con casco, aunque el detector predice dos clases opuestas. No crear dos ocupantes.'}
}
scenes=[]
for filename,spec in specs.items():
    raw=originals[filename];decision=decisions[raw['image']]
    entities=[];links=[]
    for i,index in enumerate(spec['vehicles']):
        entities.append({'id':f'm{i}','kind':'motorcycle','bbox':None,'context_bbox':raw['annotations'][index]['bbox'],
                         'origin':{'source_annotation_index':index,'class':raw['annotations'][index]['class']},'review_status':'pending'})
    for i,(index,helmet,moto) in enumerate(spec['heads']):
        entities.append({'id':f'h{i}','kind':'head','bbox':raw['annotations'][index]['bbox'],'helmet':helmet,
                         'origin':{'source_annotation_index':index,'class':raw['annotations'][index]['class']},'review_status':'pending'})
        entities.append({'id':f'o{i}','kind':'occupant','bbox':None,'role':'unknown','review_status':'pending'})
        links.append({'head_id':f'h{i}','person_id':f'o{i}','motorcycle_id':f'm{moto}' if moto is not None else None,
                      'status':'pending' if moto is not None else 'uncertain'})
    if filename=='Image_01720.jpg':
        # Bounding box comes from OUR from-scratch detector. Semantic interpretation is a review proposal.
        predictions=[json.loads(l) for l in Path('reports/error-review/detector-validation.jsonl').read_text().splitlines()]
        prediction=next(r for r in predictions if r['image']==raw['image'])['predictions'][1]
        entities += [{'id':'h_background','kind':'head','bbox':prediction['box'],'helmet':'sin_casco','review_status':'pending','origin':{'type':'own_detector_proposal','prediction_index':1}},
                     {'id':'p_background','kind':'non_occupant','bbox':None,'role':'standing_person','review_status':'pending'}]
        links.append({'head_id':'h_background','person_id':'p_background','motorcycle_id':None,'status':'pending'})
    scenes.append({'image_id':Path(filename).stem,'image':raw['image'],'sha256':raw['sha256'],'width':raw['width'],'height':raw['height'],
                   'split':decision['split'],'group_id':decision['groupId'],'note':spec['note'],'entities':entities,'links':links,'reviewed':False})
batch={'schema_version':1,'batch_id':'relations-pilot-v2','provenance':'assistant_visual_review_draft_with_existing_boxes_and_own_detector_proposals',
       'human_reviewed':False,'training_ready':False,'held_out_usage':'All images originated in v1 validation. Keep their existing split and group; do not move into training.',
       'scenes':scenes}
(ROOT/'draft.json').write_text(json.dumps(batch,ensure_ascii=False,indent=2),encoding='utf-8')
render=json.loads(json.dumps(batch))
for scene in render['scenes']:
    with Image.open(scene['image']) as im:
        im=im.convert('RGB');im.thumbnail((1200,1200));buf=io.BytesIO();im.save(buf,format='JPEG',quality=90)
    scene['preview']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
template=Path('training/annotation-review.html').read_text(encoding='utf-8')
html=template.replace('/*BATCH_DATA*/',json.dumps(render,ensure_ascii=False).replace('<','\\u003c'))
(ROOT/'review.html').write_text(html,encoding='utf-8')
print(json.dumps({'scenes':len(scenes),'entities':sum(len(s['entities']) for s in scenes),'links':sum(len(s['links']) for s in scenes),'training_ready':False,'output':str(ROOT)}))
