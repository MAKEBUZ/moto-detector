"""Conservative geometric links for separate motorcycle, person and head boxes."""
from PIL import ImageDraw


def suppress_nested_bikes(predictions):
    """Remove a weak inner box only when both boxes share a single occupant."""
    people = [p["box"] for p in predictions if p["class_id"] == 1]
    bikes = sorted((p for p in predictions if p["class_id"] == 0), key=lambda p:-p["score"])
    kept = []
    for obj in bikes:
        l,t,r,b = obj["box"]
        area = (r-l)*(b-t)
        duplicate = False
        for outer in kept:
            x,y,X,Y = outer["box"]
            inter = max(0,min(r,X)-max(l,x))*max(0,min(b,Y)-max(t,y))
            if inter < .95*area or area > .75*(X-x)*(Y-y) or outer["score"]-obj["score"] < .15:
                continue
            centers = [(a+c)/2 for a,top,c,bottom in people if x <= (a+c)/2 <= X and top < Y and bottom > y]
            if len(centers)==1 and l <= centers[0] <= r:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
    return [p for p in predictions if p["class_id"] != 0 or p in kept]


def assign_heads(riders, heads):
    from functools import lru_cache
    candidates = []
    for head in heads:
        x,y,X,Y = head["box"]
        cx,cy = (x+X)/2,(y+Y)/2
        options = []
        for i,rider in enumerate(riders):
            l,t,r,b = rider["box"]
            w,h = max(1,r-l),max(1,b-t)
            if l <= cx <= r and t-.08*h <= cy <= t+.55*h:
                cost = abs(cx-(l+r)/2)/w + .6*abs((cy-t)/h-.16)
                options.append((i,1.1-cost))
        candidates.append(options)
    if len(riders)>10 or len(heads)>14:
        for head,options in zip(heads,candidates):
            if len(options)==1:
                riders[options[0][0]]["cabezas"].append(head)
        return

    def solve(forbidden=None):
        @lru_cache(None)
        def best(j,mask):
            if j==len(heads):
                return 0., ()
            value,path = best(j+1,mask)
            for i,weight in candidates[j]:
                if mask & (1<<i) or (j,i)==forbidden:
                    continue
                tail,links = best(j+1,mask|(1<<i))
                if weight+tail>value:
                    value,path = weight+tail, ((j,i),)+links
            return value,path
        return best(0,0)

    score,links = solve()
    for j,i in links:
        # Abstain when another assignment explains the boxes nearly as well.
        if score-solve((j,i))[0] < .045:
            continue
        if any(k!=j and other["class_id"]!=heads[j]["class_id"] and
               box_overlap(other["box"],heads[j]["box"])>.4 for k,other in enumerate(heads)):
            continue
        riders[i]["cabezas"].append(heads[j])


def box_overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def summarize(predictions):
    groups = [[p for p in predictions if p['class_id'] == c] for c in range(4)]
    bikes, people, helmets, bare = groups
    riders = []
    for i, person in enumerate(people, 1):
        left, top, right, bottom = person['box']
        center = (left + right) / 2
        candidates = []
        for bike_id, bike in enumerate(bikes, 1):
            x, y, X, Y = bike['box']
            if x <= center <= X and top < (y + Y) / 2 and y <= bottom <= Y + .25 * (Y-y):
                candidates.append(bike_id)
        riders.append(dict(id=i, box=person['box'], moto=candidates[0] if len(candidates)==1 else None,
                           motos_candidatas=candidates, estado='indeterminado', cabezas=[]))
    assign_heads(riders, helmets + bare)
    for rider in riders:
        if len(rider['cabezas'])==1:
            rider['estado'] = 'con_casco' if rider['cabezas'][0]['class_id']==2 else 'sin_casco'
    counts = {state:sum(r['estado']==state for r in riders)
              for state in ('con_casco','sin_casco','indeterminado')}
    counts.update(motos=len(bikes), ocupantes=len(riders), sin_moto_asignada=sum(r['moto'] is None for r in riders))
    return dict(motos=[dict(id=i,box=p['box'],ocupantes=[r['id'] for r in riders if r['moto']==i])
                       for i,p in enumerate(bikes,1)], ocupantes=riders, counts=counts)


def annotate(image, summary, show_track_state=False):
    draw = ImageDraw.Draw(image)
    def label(box, text, color):
        x,y = box[:2]
        draw.rectangle(box, outline=color, width=3)
        bounds=draw.textbbox((x,y),text)
        draw.rectangle(bounds,fill='black')
        draw.text((x,y),text,fill=color)
    for bike in summary['motos']:
        label(bike['box'], f"Moto {bike['id']}", 'lime')
    for r in summary['ocupantes']:
        moto = f"M{r['moto']}" if r['moto'] else 'moto ?'
        caption = f"P{r['id']} / {moto}"
        if show_track_state:
            caption += " / " + r["estado"].replace("_", " ")
        label(r['box'], caption, 'yellow')
        for head in r['cabezas']:
            color = 'cyan' if head['class_id']==2 else 'orange'
            state = r["estado"].replace("_", " ")
            if show_track_state and r["estado"] == "indeterminado" and head["class_id"] == 2:
                state = "casco visible"
            label(head['box'], f"P{r['id']} {state}", color)
