import json
from pathlib import Path
from collections import Counter
from review_errors import match

out=Path('reports/error-review')
rows=[json.loads(l) for l in (out/'detector-validation.jsonl').read_text().splitlines()]
counts=Counter();confusion=[[0,0],[0,0]];examples=[]
for r in rows:
    heads=[p for p in r['predictions'] if p['class_id'] in (1,2)]
    truth=[g for g in r['ground_truth'] if g['class_id'] in (1,2)]
    pairs,fp,fn=match([{**p,'class_id':1} for p in heads],[{**g,'class_id':1} for g in truth],.5)
    counts['ground_truth_heads']+=len(truth);counts['localized_heads']+=len(pairs);counts['unmatched_predictions']+=len(fp);counts['missed_heads']+=len(fn)
    for i,j in pairs:
        confusion[truth[j]['class_id']-1][heads[i]['class_id']-1]+=1
        if heads[i]['class_id']!=truth[j]['class_id']:
            counts['wrong_helmet_class_after_localization']+=1
            examples.append({'image_id':r['id'],'prediction':heads[i],'ground_truth':truth[j]})
result={**dict(counts),'class_agnostic_head_recall':counts['localized_heads']/counts['ground_truth_heads'],'confusion_among_localized_heads_true_rows':confusion,'threshold':.5,'iou':.5,'test_used':False}
(out/'localization-vs-classification.json').write_text(json.dumps(result,indent=2));(out/'localized-head-class-errors.json').write_text(json.dumps(examples,indent=2));print(json.dumps(result))
