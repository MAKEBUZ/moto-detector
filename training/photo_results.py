"""Conservative geometric links for separate motorcycle, person and head boxes."""
from PIL import ImageDraw


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
    ambiguous = set()
    for head in helmets + bare:
        x, y, X, Y = head['box']
        cx, cy = (x+X)/2, (y+Y)/2
        owners = []
        for rider in riders:
            l, t, r, b = rider['box']
            if l <= cx <= r and t <= cy <= t+.55*(b-t):
                owners.append(rider)
        if len(owners)==1:
            owners[0]['cabezas'].append(head)
        else:
            ambiguous.update(owner['id'] for owner in owners)
    for rider in riders:
        if rider['id'] not in ambiguous and len(rider['cabezas'])==1:
            rider['estado'] = 'con_casco' if rider['cabezas'][0]['class_id']==2 else 'sin_casco'
    counts = {state:sum(r['estado']==state for r in riders)
              for state in ('con_casco','sin_casco','indeterminado')}
    counts.update(motos=len(bikes), ocupantes=len(riders), sin_moto_asignada=sum(r['moto'] is None for r in riders))
    return dict(motos=[dict(id=i,box=p['box'],ocupantes=[r['id'] for r in riders if r['moto']==i])
                       for i,p in enumerate(bikes,1)], ocupantes=riders, counts=counts)


def annotate(image, summary):
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
        label(r['box'], f"P{r['id']} / {moto} / {r['estado'].replace('_',' ')}", 'yellow')
