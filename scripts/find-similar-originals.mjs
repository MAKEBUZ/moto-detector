// Candidate retrieval only. dHash similarity is not proof of image/scene identity.
import fs from 'node:fs';
const jsonl=f=>fs.readFileSync(f,'utf8').replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
const records=jsonl('reports/roboflow-manifest.jsonl');
const decoded=new Map(jsonl('reports/roboflow-image-decoding.jsonl').map(r=>[r.image,r]));
if(fs.existsSync('reports/edgevision-manifest.jsonl'))for(const r of jsonl('reports/edgevision-manifest.jsonl')){records.push(r);decoded.set(r.image,r);}
const valid=records.filter(r=>decoded.get(r.image)?.decoded&&decoded.get(r.image)?.dhash);
const high=valid.map(r=>parseInt(decoded.get(r.image).dhash.slice(0,8),16)>>>0);
const low=valid.map(r=>parseInt(decoded.get(r.image).dhash.slice(8),16)>>>0);
function pop(x){x=x-((x>>>1)&0x55555555);x=(x&0x33333333)+((x>>>2)&0x33333333);return (((x+(x>>>4))&0x0f0f0f0f)*0x01010101)>>>24;}
const parent=valid.map((_,i)=>i);
const root=i=>{while(parent[i]!==i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
const examples=[];let pairs=0,crossSplitPairs=0,crossSourcePairs=0;
for(let i=0;i<valid.length;i++)for(let j=i+1;j<valid.length;j++){
 const d=pop(high[i]^high[j])+pop(low[i]^low[j]);
 if(d>4)continue;
 pairs++;
 if(valid[i].upstreamSplit!==valid[j].upstreamSplit)crossSplitPairs++;
 if(valid[i].source!==valid[j].source)crossSourcePairs++;
 parent[root(j)]=root(i);
 if(examples.length<200)examples.push({a:valid[i].image,b:valid[j].image,distance:d});
}
const groups=new Map();
valid.forEach((r,i)=>{const k=root(i),g=groups.get(k)??[];g.push({image:r.image,source:r.source,split:r.upstreamSplit,dhash:decoded.get(r.image).dhash});groups.set(k,g);});
const candidates=[...groups.values()].filter(g=>g.length>1);
const exact=new Map();for(const r of records){const g=exact.get(r.sha256)??[];g.push({image:r.image,source:r.source,split:r.upstreamSplit});exact.set(r.sha256,g);}
const exactGroups=[...exact.values()].filter(g=>g.length>1);
const report={generatedAt:new Date().toISOString(),method:'64-bit dHash, Hamming distance <= 4; connected components are REVIEW CANDIDATES, not confirmed duplicates or final split groups.',imagesCompared:valid.length,pairs,crossSplitPairs,crossSourcePairs,groups:candidates,examplePairs:examples,exactDuplicateGroups:exactGroups,exactCrossSplitGroups:exactGroups.filter(g=>new Set(g.map(r=>r.split)).size>1)};
fs.writeFileSync('reports/original-similarity-candidates.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({imagesCompared:valid.length,pairs,crossSplitPairs,crossSourcePairs,candidateGroups:candidates.length,exactGroups:exactGroups.length,exactCrossSplitGroups:report.exactCrossSplitGroups.length}));
