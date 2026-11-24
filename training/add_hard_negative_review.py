"""Add pending model-region review to an existing source-label annotation batch."""
import json
from pathlib import Path


def main():
    folder=Path('data/annotations/train-hard-negatives-v2-01')
    draft=folder/'draft.json';batch=json.loads(draft.read_text(encoding='utf-8'))
    if any('hard_negative_candidates' in s for s in batch['scenes']):raise FileExistsError('Already enriched')
    mined=json.loads(Path('reports/separate-v2/hard-negatives-selection/visual-candidates.json').read_text())
    for s in batch['scenes']:
        r=next(r for r in mined if r['image_id']==s['image_id'])
        assert r['group_id']==s['group_id'] and s['split']=='train'
        s['hard_negative_candidates']=[{'id':'n0','bbox':r['box'],'predicted_label':r['label'],'score':r['score'],
            'checkpoint':r['run'],'checkpoint_sha256':r['checkpoint_sha256'],'decision':'pending','category':'pending'}]
        s['note']+=' NEGATIVO DIFÍCIL: '+s['selection_reason']+' La caja magenta n0 es solo una sugerencia del modelo, no una etiqueta. No dibujar asiento/ropa/piezas como cabeza. Revisar la imagen completa.'
    batch['hard_negative_policy']='Review hints only; no fifth class. All true objects must remain labeled. Pending/uncertain/real-head candidates require further review before preparation.'
    draft.write_text(json.dumps(batch,ensure_ascii=False,indent=2),encoding='utf-8')
    html=(folder/'review.html').read_text(encoding='utf-8')
    start=html.index('const original=')+len('const original=');end=html.index(';\nlet batch=',start)
    embedded=json.loads(html[start:end])
    for a,b in zip(batch['scenes'],embedded['scenes']):a['preview']=b['preview']
    html=html[:start]+json.dumps(batch,ensure_ascii=False).replace('<','\\u003c')+html[end:]
    html=html.replace('<h3>1. Entidad y caja</h3>','<h3>0. Regiones confundidas con cabeza</h3><p>Magenta: sugerencia del modelo. Decide si es asiento, ropa u otra pieza; no crees una clase nueva. Si es cabeza real o hay duda, deja cobertura completa desmarcada y solicita revisión.</p><div id="negative-review"></div><h3>1. Entidad y caja</h3>')
    extension=r'''
const baseDraw=draw;
draw=function(){baseDraw();const sx=canvas.width/scene().width,sy=canvas.height/scene().height;
 for(const r of scene().hard_negative_candidates||[]){const b=r.bbox;ctx.strokeStyle='#e000c0';ctx.lineWidth=3;ctx.setLineDash([4,4]);ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);ctx.fillStyle='#e000c0';ctx.fillText(r.id+' revisar',b[0]*sx,Math.max(18,b[1]*sy));}ctx.setLineDash([]);};
const baseRender=render;
render=function(){baseRender();const el=$('negative-review');el.replaceChildren();
 for(const r of scene().hard_negative_candidates||[]){const row=document.createElement('div');const label=document.createElement('p');label.textContent=r.id+' · confianza del modelo '+(100*r.score).toFixed(1)+'% (no prueba de acierto)';row.append(label);
 const decisions=document.createElement('select');for(const [v,name] of [['pending','Pendiente'],['background','No es cabeza'],['real_head','Es una cabeza real'],['uncertain','No puedo determinarlo']])decisions.append(option(v,name,r.decision));
 decisions.onchange=()=>{r.decision=decisions.value;scene().reviewed=false;if(r.decision!=='background')scene().annotation_complete=false;render();};row.append(decisions);
 const categories=document.createElement('select');for(const [v,name] of [['pending','Tipo pendiente'],['seat','Asiento'],['clothing','Ropa / mochila'],['motorcycle_part','Pieza de moto / espejo'],['other','Otra región']])categories.append(option(v,name,r.category));
 categories.onchange=()=>{r.category=categories.value;scene().reviewed=false;render();};row.append(categories);el.append(row);}};
const baseErrors=errors;
errors=function(){const list=baseErrors();for(const s of batch.scenes){for(const r of s.hard_negative_candidates||[]){if(r.decision==='pending')list.push(s.image_id+': región negativa pendiente.');if(r.decision==='background'&&r.category==='pending')list.push(s.image_id+': falta tipo de región.');if(r.decision!=='background'&&s.annotation_complete)list.push(s.image_id+': región dudosa/cabeza real; desmarca cobertura completa y solicita revisión.');}}return list;};
render();
'''
    html=html.replace('</script>',extension+'\n</script>')
    (folder/'review.html').write_text(html,encoding='utf-8')
    print('Six pending regions added; annotations still unconfirmed.')


if __name__=='__main__':main()
