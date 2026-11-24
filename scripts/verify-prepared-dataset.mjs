import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {digest,validateBox,pixelBox,chooseSplits} from './dataset-utils.mjs';
const base='data/prepared/helmet-baseline-v1';
const read=f=>fs.readFileSync(f,'utf8').replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
const det=read(`${base}/detector-manifest.jsonl`),crops=read(`${base}/classifier-manifest.jsonl`),exported=read(`${base}/classifier-export.jsonl`);
const decisions=read('reports/preparation/image-decisions.jsonl'),originals=new Map([...read('reports/edgevision-manifest.jsonl'),...read('reports/roboflow-manifest.jsonl')].map(r=>[r.image,r]));
const dims=new Map(read('reports/roboflow-image-decoding.jsonl').map(r=>[r.image,r]));
const groups=JSON.parse(fs.readFileSync(`${base}/groups.json`,'utf8'));
const contract=JSON.parse(fs.readFileSync(`${base}/training-contract.json`,'utf8'));
assert.equal(contract.pretrained,false);assert.equal(contract.initialization,'random');assert.equal(contract.externalWeightsAllowed,false);
const assigned=chooseSplits(groups,contract.seed);for(const g of groups)assert.equal(assigned.get(g.id),g.split,'Nonreproducible split assignment');
assert.equal(decisions.length,13303);assert.equal(new Set(decisions.map(r=>r.image)).size,13303);
const rowsByImage=new Map(decisions.map(r=>[r.image,r]));
const exports=new Map(exported.map(r=>[r.id,r]));assert.equal(exports.size,crops.length);assert.equal(exported.length,crops.length);
assert.equal(new Set(det.map(r=>r.id)).size,det.length);assert.equal(new Set(crops.map(r=>r.id)).size,crops.length);
const imageSplits=new Map(),groupSplits=new Map(),rawHashes=new Map();
function register(r){
 assert.equal(r.split,assigned.get(r.groupId));
 for(const [map,key]of [[imageSplits,r.image],[groupSplits,r.groupId]]){if(map.has(key))assert.equal(map.get(key),r.split,'Leakage between tasks or splits');else map.set(key,r.split);}
 if(!rawHashes.has(r.image))rawHashes.set(r.image,digest(fs.readFileSync(r.image)));
 assert.equal(rawHashes.get(r.image),originals.get(r.image).sha256,'Raw input changed');
}
const detectorHashes=new Map();
for(const r of det){
 register(r);assert.equal(originals.get(r.image).source,'edgevision');assert.equal(rowsByImage.get(r.image).detector,true);
 const file=`${base}/detector/images/${r.split}/${r.id}.jpg`,label=`${base}/detector/labels/${r.split}/${r.id}.txt`;
 const hash=digest(fs.readFileSync(file));assert.equal(hash,r.sha256,'Detector copy differs from preserved source');
 assert.ok(!detectorHashes.has(hash),'Repeated full image in detector');detectorHashes.set(hash,r.split);
 const lines=fs.readFileSync(label,'utf8').trim().split(/\r?\n/).map(l=>l.split(/\s+/).map(Number));assert.equal(lines.length,r.annotations.length);
 for(const [i,v]of lines.entries()){
  assert.equal(v.length,5);assert.ok(v.every(Number.isFinite));assert.equal(v[0],r.annotations[i].classId);
  const [x0,y0,x1,y1]=r.annotations[i].box;assert.equal(validateBox([x0,y0,x1,y1],r.width,r.height),null);
  const orig=originals.get(r.image).annotations[r.annotations[i].index];assert.deepEqual(orig.bbox,r.annotations[i].box);assert.equal({BikeWithRider:0,Helmet:1,NoHelmet:2}[orig.class],v[0]);
  const expected=[(x0+x1)/2/r.width,(y0+y1)/2/r.height,(x1-x0)/r.width,(y1-y0)/r.height];
  v.slice(1).forEach((n,j)=>assert.ok(Math.abs(n-expected[j])<1e-7));
 }
}
function jpegSize(b){let i=2;assert.equal(b.readUInt16BE(0),0xffd8);while(i+8<b.length){assert.equal(b[i],255);while(b[i]===255)i++;const m=b[i++];if(m===217||m===218)break;if(m===1||(m>=208&&m<=215))continue;const n=b.readUInt16BE(i);if([192,193,194].includes(m))return [b.readUInt16BE(i+5),b.readUInt16BE(i+3)];i+=n;}throw Error('Unreadable JPEG size');}
const cropHashes=new Map();
for(const r of crops){
 register(r);const e=exports.get(r.id);assert.ok(e);assert.equal(e.path,`${base}/classifier/${r.split}/${r.className}/${r.id}.jpg`);
 const raw=originals.get(r.image),size=raw.source==='edgevision'?raw:dims.get(r.image),a=raw.annotations[r.annotationIndex];
 assert.equal(raw.source,'edgevision','Unreviewed source in the approved baseline');
 assert.equal({Helmet:0,NoHelmet:1,'With Helmet':0,'Without Helmet':1}[a.class],r.classId);assert.equal(['con_casco','sin_casco'][r.classId],r.className);
 const box=pixelBox(a,size.width,size.height);assert.equal(validateBox(box,size.width,size.height),null);assert.deepEqual(r.bounds,[Math.floor(box[0]),Math.floor(box[1]),Math.ceil(box[2]),Math.ceil(box[3])]);
 const bytes=fs.readFileSync(e.path),sha=digest(bytes);assert.equal(sha,e.sha256);assert.deepEqual(jpegSize(bytes),[e.width,e.height]);assert.ok(Math.min(e.width,e.height)>=24);
 const group=cropHashes.get(sha)??[];group.push({id:r.id,split:r.split,classId:r.classId});cropHashes.set(sha,group);
}
const duplicates=[...cropHashes.values()].filter(g=>g.length>1),leaks=duplicates.filter(g=>new Set(g.map(r=>r.split)).size>1),conflicts=duplicates.filter(g=>new Set(g.map(r=>r.classId)).size>1);
fs.writeFileSync('reports/preparation/crop-duplicates.json',JSON.stringify({duplicates,leaks,conflicts},null,2)+'\n');
assert.equal(leaks.length,0,'Identical crops across splits');assert.equal(conflicts.length,0,'Identical crops with conflicting classes');
for(const relation of JSON.parse(fs.readFileSync('reports/preparation/group-relations.json','utf8'))){const splits=new Set(relation.members.filter(m=>imageSplits.has(m)).map(m=>imageSplits.get(m)));assert.ok(splits.size<=1,'Known relation crosses split');}
for(const split of ['train','val','test']){
 const count=det.filter(r=>r.split===split).length;assert.equal(fs.readdirSync(`${base}/detector/images/${split}`).length,count);assert.equal(fs.readdirSync(`${base}/detector/labels/${split}`).length,count);
 for(const className of ['con_casco','sin_casco'])assert.equal(fs.readdirSync(`${base}/classifier/${split}/${className}`).length,crops.filter(r=>r.split===split&&r.className===className).length);
}
const result={status:'passed',scope:'Ready for baseline training, not certified scene-independent evaluation or separate rider detection.',detectorImages:det.length,classifierCrops:crops.length,sourceImagesUsed:imageSplits.size,groupsUsed:groupSplits.size,exactCropDuplicateGroups:duplicates.length,crossSplitExactCropDuplicates:leaks.length,knownRelationsCrossingSplits:0,verifiedAtUtc:new Date().toISOString()};
fs.writeFileSync('reports/preparation/verification.json',JSON.stringify(result,null,2)+'\n');
const fingerprintFiles=['data/sources.lock.json','reports/edgevision-manifest.jsonl','reports/roboflow-manifest.jsonl','reports/roboflow-image-decoding.jsonl','reports/original-similarity-candidates.json','scripts/prepare-dataset.mjs','scripts/dataset-utils.mjs','scripts/export-classifier-crops.ps1',`${base}/detector-manifest.jsonl`,`${base}/classifier-manifest.jsonl`,`${base}/classifier-export.jsonl`,`${base}/groups.json`,`${base}/training-contract.json`];
const lock={version:'helmet-baseline-v1',seed:contract.seed,inputsAndOutputs:Object.fromEntries(fingerprintFiles.map(f=>[f,digest(fs.readFileSync(f))])),verifiedAtUtc:result.verifiedAtUtc};
fs.writeFileSync('reports/preparation/dataset-lock.json',JSON.stringify(lock,null,2)+'\n');
fs.writeFileSync(`${base}/dataset-lock.json`,JSON.stringify(lock,null,2)+'\n');
const summary=JSON.parse(fs.readFileSync('reports/preparation/summary.json','utf8'));summary.status='ready_for_baseline_training';fs.writeFileSync('reports/preparation/summary.json',JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
