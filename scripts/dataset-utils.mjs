import crypto from 'node:crypto';
export const digest=s=>crypto.createHash('sha256').update(s).digest('hex');
export function validateBox(box,width,height){
 if(box.length!==4||![...box,width,height].every(Number.isFinite)||width<=0||height<=0)return 'invalid_numbers';
 const [x0,y0,x1,y1]=box;
 if(x1<=x0||y1<=y0)return 'degenerate_box';
 if(x0<0||y0<0||x1>width||y1>height)return 'outside_image';
 return null;
}
export function pixelBox(a,width,height){
 if(a.bbox)return a.bbox;
 const b=a.coordinates;
 if(!b||b.length!==4)return null;
 return [(b[0]-b[2]/2)*width,(b[1]-b[3]/2)*height,(b[0]+b[2]/2)*width,(b[1]+b[3]/2)*height];
}
export class UnionFind{
 constructor(keys){this.parent=new Map(keys.map(k=>[k,k]));}
 root(k){if(!this.parent.has(k))throw Error(`Unknown group member: ${k}`);let r=k;while(this.parent.get(r)!==r)r=this.parent.get(r);while(this.parent.get(k)!==k){const n=this.parent.get(k);this.parent.set(k,r);k=n;}return r;}
 join(a,b){const x=this.root(a),y=this.root(b);if(x!==y)this.parent.set(x<y?y:x,x<y?x:y);}
}
export function chooseSplits(groups,seed){
 // Greedy grouped allocation balanced by images and class/source counts.
 const ratios={train:.7,val:.15,test:.15};const splits=Object.keys(ratios);
 const totals={};for(const g of groups)for(const [k,v] of Object.entries(g.features))totals[k]=(totals[k]??0)+v;
 const current=Object.fromEntries(splits.map(s=>[s,{}]));const allocation=new Map();
 const sorted=[...groups].sort((a,b)=>b.size-a.size||digest(seed+a.id).localeCompare(digest(seed+b.id)));
 for(const g of sorted){
  let best=null,bestCost=Infinity;
  for(const s of splits){let cost=0;for(const [k,v] of Object.entries(g.features)){const target=totals[k]*ratios[s];const before=current[s][k]??0;cost+=((before+v-target)**2-(before-target)**2)/Math.max(1,totals[k]);}
   if(cost<bestCost){bestCost=cost;best=s;}}
  allocation.set(g.id,best);for(const [k,v] of Object.entries(g.features))current[best][k]=(current[best][k]??0)+v;
 }
 return allocation;
}
