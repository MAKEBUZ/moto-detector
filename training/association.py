"""Geometric candidates, not proof that a head belongs to a motorcycle occupant."""
def iou(a, b):
    area = lambda x: max(0, x[2]-x[0])*max(0, x[3]-x[1])
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection / max(1e-9, area(a)+area(b)-intersection)


def associate(predictions, threshold=.5, margin=.15):
    motorcycles = [(i,p) for i,p in enumerate(predictions) if p['class_id']==0 and p['score']>=threshold]
    heads = sorted([(i,p) for i,p in enumerate(predictions) if p['class_id'] in (1,2) and p['score']>=threshold], key=lambda x:-x[1]['score'])
    results=[]; retained=[]
    for index,head in heads:
        duplicates=[(j,p) for j,p in retained if iou(head['box'],p['box'])>=.6]
        if duplicates:
            conflict=any(p['class_id']!=head['class_id'] for _,p in duplicates)
            results.append({'head_index':index,'motorcycle_index':None,'status':'conflicting_head' if conflict else 'duplicate_head','candidates':[]})
            if conflict:
                for previous in results:
                    if previous['head_index'] in [j for j,_ in duplicates]:
                        previous['status']='conflicting_head';previous['motorcycle_index']=None
            continue
        retained.append((index,head))
        x0,y0,x1,y1=head['box']; cx=(x0+x1)/2;cy=(y0+y1)/2
        candidates=[]
        for mi,moto in motorcycles:
            a,b,c,d=moto['box'];w=c-a;h=d-b
            if min(w,h,x1-x0,y1-y0)<=0:continue
            intersection=max(0,min(x1,c)-max(x0,a))*max(0,min(y1,d)-max(y0,b))
            contained=intersection/((x1-x0)*(y1-y0))
            if contained<.8 or not (a<=cx<=c and b<=cy<=b+.6*h):continue
            if not (.025<=(y1-y0)/h<=.4 and (x1-x0)/w<=.65):continue
            cost=abs(cx-(a+c)/2)/w+abs((cy-b)/h-.2)*.5
            candidates.append({'motorcycle_index':mi,'cost':round(cost,6),'head_containment':round(contained,6)})
        candidates.sort(key=lambda x:(x['cost'],x['motorcycle_index']))
        status='unassigned';selected=None
        if candidates:
            if len(candidates)>1 and candidates[1]['cost']-candidates[0]['cost']<margin:status='ambiguous'
            else:status='candidate_requires_review';selected=candidates[0]['motorcycle_index']
        results.append({'head_index':index,'motorcycle_index':selected,'status':status,'candidates':candidates})
    return {'associations':results,'occupancy_confirmed':False,'counting_allowed':False,
            'limitation':'Geometry cannot distinguish an occupant from a pedestrian behind or beside the motorcycle. No driver/passenger roles inferred.'}
