import fs from 'node:fs';
import path from 'node:path';
import {digest,validateBox,pixelBox,UnionFind,chooseSplits} from './dataset-utils.mjs';
const read=f=>fs.readFileSync(f,'utf8').replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
const rows=[...read('reports/edgevision-manifest.jsonl'),...read('reports/roboflow-manifest.jsonl')].sort((a,b)=>a.image.localeCompare(b.image));
const dims=new Map(read('reports/roboflow-image-decoding.jsonl').map(r=>[r.image,r]));
const similarities=JSON.parse(fs.readFileSync('reports/original-similarity-candidates.json','utf8'));
const out='data/prepared/helmet-baseline-v1';
if(fs.existsSync(out))throw Error(`${out} already exists. Preserve published versions; use a new version for changed policies.`);
const seed='helmet-baseline-v1-2026-11-23';
const uf=new UnionFind(rows.map(r=>r.image));const relations=[];
function joinGroup(members,reason){if(members.length<2)return;for(const m of members.slice(1))uf.join(members[0],m);relations.push({reason,members});}
for(const g of similarities.exactDuplicateGroups)joinGroup(g.map(r=>r.image),'exact_sha256');
// Conservative co-location, not a claim that perceptual candidates are duplicates.
for(const g of similarities.groups)joinGroup(g.map(r=>r.image),'perceptual_candidate_keep_together');
const lineage=new Map();
for(const r of rows){
 const stem=path.parse(r.image).name.replace(/\.rf\.[a-f0-9]+$/i,'');
 const key=r.groupId?`video:${r.groupId}`:`lineage:${r.source}:${stem}`;
 const g=lineage.get(key)??[];g.push(r.image);lineage.set(key,g);
}
for(const [key,g]of lineage)joinGroup(g,key.startsWith('video:')?'video_name':'export_lineage');
// EdgeVision lacks scene IDs. Adjacent filename blocks add a conservative guard;
// these are NOT verified scene labels. Similarity can join blocks further.
const blocks=new Map();
for(const r of rows.filter(r=>r.source==='edgevision')){
 const number=Number(path.parse(r.image).name.match(/(\d+)$/)?.[1]);
 if(!Number.isInteger(number))throw Error('Unexpected EdgeVision name');
 const key=Math.floor((number-1)/25);const g=blocks.get(key)??[];g.push(r.image);blocks.set(key,g);
}
for(const g of blocks.values())joinGroup(g,'edgevision_adjacent_filename_block_25_not_verified_scene');
const members=new Map();for(const r of rows){const root=uf.root(r.image),g=members.get(root)??[];g.push(r.image);members.set(root,g);}
const ids=new Map([...members].map(([root,g])=>[root,digest(g.sort().join('\n')).slice(0,20)]));
const dispositions=[],crops=[],detector=[],corrections=[];
const labelMap={Helmet:0,NoHelmet:1,'With Helmet':0,'Without Helmet':1};
const duplicateImages=new Set();for(const g of similarities.exactDuplicateGroups)for(const r of [...g].sort((a,b)=>a.image.localeCompare(b.image)).slice(1))duplicateImages.add(r.image);
for(const r of rows){
 const d=r.source==='edgevision'?r:dims.get(r.image);const groupId=ids.get(uf.root(r.image));
 const item={image:r.image,sha256:r.sha256,source:r.source,version:r.version,groupId,upstreamSplit:r.upstreamSplit,detector:false,classifierCrops:0,reasons:[]};
 dispositions.push(item);
 if(duplicateImages.has(r.image)){item.reasons.push('exact_duplicate_not_selected');continue;}
 if(r.source==='motorcycle_helmet_od'){item.reasons.push('mosaic_lineage_and_head_labels_require_review');continue;}
 if(r.source!=='edgevision'){item.reasons.push('source_requires_semantic_reannotation');continue;}
 if(!d?.decoded||!r.unifiedLabelsMatch){item.reasons.push('failed_input_validation');continue;}
 if(!r.annotations.length){item.reasons.push('unverified_empty_annotation');continue;}
 const boxes=[];let invalid=false;const seen=new Set();
 for(const [index,a]of r.annotations.entries()){
  const box=pixelBox(a,d.width,d.height);const reason=box?validateBox(box,d.width,d.height):'unsupported_geometry';
  if(reason){item.reasons.push(reason);invalid=true;continue;}
  const key=JSON.stringify([a.class,...box]);if(seen.has(key)){corrections.push({image:r.image,index,action:'remove_exact_repeated_annotation'});continue;}seen.add(key);
  boxes.push({class:a.class,box,index});
 }
 // Quarantine the entire image: dropping a bad box alone would create unlabeled objects.
 if(invalid){item.reasons=[...new Set(item.reasons)];continue;}
 const headConflict=(a,b)=>{
  const ac=labelMap[a.class],bc=labelMap[b.class];if(ac===undefined||bc===undefined||ac===bc)return false;
  const [x0,y0,x1,y1]=a.box,[u0,v0,u1,v1]=b.box;
  const intersection=Math.max(0,Math.min(x1,u1)-Math.max(x0,u0))*Math.max(0,Math.min(y1,v1)-Math.max(y0,v0));
  return intersection/Math.min((x1-x0)*(y1-y0),(u1-u0)*(v1-v0))>.5;
 };
 const detectorConflict=boxes.some((a,i)=>boxes.slice(i+1).some(b=>headConflict(a,b)));
 if(r.source==='edgevision'&&detectorConflict)item.reasons.push('detector_conflicting_head_labels');
 if(r.source==='edgevision'&&!detectorConflict){
  const mapped=boxes.map(a=>({classId:{BikeWithRider:0,Helmet:1,NoHelmet:2}[a.class],box:a.box,index:a.index}));
  if(mapped.some(a=>a.classId===undefined))throw Error('Unexpected EdgeVision class');
  item.detector=true;detector.push({id:digest(r.image).slice(0,20),image:r.image,sha256:r.sha256,groupId,width:d.width,height:d.height,annotations:mapped});
 }
 for(const a of boxes){
  const cls=labelMap[a.class];if(cls===undefined)continue;
  const [x0,y0,x1,y1]=a.box;
  const bounds=[Math.floor(x0),Math.floor(y0),Math.ceil(x1),Math.ceil(y1)];
  if(Math.min(x1-x0,y1-y0)<24){corrections.push({image:r.image,index:a.index,action:'exclude_classifier_crop_below_24px'});continue;}
  // A conflicting, strongly overlapping head label needs review rather than majority vote.
  const conflict=boxes.some(b=>headConflict(a,b));
  if(conflict){corrections.push({image:r.image,index:a.index,action:'exclude_conflicting_head_crop'});continue;}
  crops.push({id:digest(`${r.image}#${a.index}`).slice(0,24),image:r.image,source:r.source,sourceSha256:r.sha256,annotationIndex:a.index,groupId,classId:cls,className:['con_casco','sin_casco'][cls],bounds});item.classifierCrops++;
 }
 if(!item.detector&&!item.classifierCrops)item.reasons.push('no_eligible_crops');
}
const groups=new Map();
function add(groupId,key,n=1){const g=groups.get(groupId)??{id:groupId,size:0,features:{}};g.features[key]=(g.features[key]??0)+n;groups.set(groupId,g);}
for(const r of dispositions.filter(r=>r.detector||r.classifierCrops)){add(r.groupId,`source:${r.source}`);groups.get(r.groupId).size++;}
for(const r of detector){add(r.groupId,'detector_images');for(const a of r.annotations)add(r.groupId,`detector_class:${a.classId}`);}
for(const r of crops)add(r.groupId,`classifier:${r.source}:${r.classId}`);
const allocations=chooseSplits([...groups.values()],seed);
for(const r of [...dispositions,...detector,...crops])r.split=allocations.get(r.groupId)??null;
for(const split of ['train','val','test']){
 if(!detector.some(r=>r.split===split))throw Error(`No detector examples for ${split}; grouping too restrictive`);
 for(const cls of [0,1])if(!crops.some(r=>r.split===split&&r.classId===cls))throw Error('Missing classifier class in split');
}
fs.mkdirSync(out,{recursive:true});fs.mkdirSync('reports/preparation',{recursive:true});
function jsonl(file,data){fs.writeFileSync(file,data.map(r=>JSON.stringify(r)).join('\n')+'\n');}
jsonl(`${out}/detector-manifest.jsonl`,detector);jsonl(`${out}/classifier-manifest.jsonl`,crops);
jsonl('reports/preparation/image-decisions.jsonl',dispositions);jsonl('reports/preparation/annotation-decisions.jsonl',corrections);
jsonl('reports/preparation/review-queue.jsonl',dispositions.filter(r=>r.reasons.some(reason=>reason!=='exact_duplicate_not_selected')));
fs.writeFileSync('reports/preparation/group-relations.json',JSON.stringify(relations,null,2)+'\n');
fs.writeFileSync(`${out}/groups.json`,JSON.stringify([...groups.values()].map(g=>({...g,split:allocations.get(g.id)})),null,2)+'\n');
for(const r of detector){
 const imgDir=`${out}/detector/images/${r.split}`,lblDir=`${out}/detector/labels/${r.split}`;fs.mkdirSync(imgDir,{recursive:true});fs.mkdirSync(lblDir,{recursive:true});
 // Copies keep prepared images independent from the preserved raw files.
 fs.copyFileSync(r.image,`${imgDir}/${r.id}.jpg`);
 const lines=r.annotations.map(a=>{const [x0,y0,x1,y1]=a.box;return [a.classId,((x0+x1)/2/r.width).toFixed(8),((y0+y1)/2/r.height).toFixed(8),((x1-x0)/r.width).toFixed(8),((y1-y0)/r.height).toFixed(8)].join(' ');});
 fs.writeFileSync(`${lblDir}/${r.id}.txt`,lines.join('\n')+'\n');
}
fs.writeFileSync(`${out}/detector/data.yaml`,`path: ${JSON.stringify(path.resolve(out,'detector').replaceAll('\\','/'))}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 3\nnames: ['moto_con_ocupante', 'cabeza_con_casco', 'cabeza_sin_casco']\n`);
const stats={seed,inputImages:rows.length,detectorImages:detector.length,classifierCrops:crops.length,eligibleGroups:groups.size,largestGroup:Math.max(...[...groups.values()].map(g=>g.size)),quarantinedImages:dispositions.filter(r=>!r.detector&&!r.classifierCrops).length,splits:{},reasonCounts:{},correctionCounts:{},status:'awaiting_crop_export_and_verification'};
for(const r of dispositions)for(const reason of r.reasons)stats.reasonCounts[reason]=(stats.reasonCounts[reason]??0)+1;
for(const r of corrections)stats.correctionCounts[r.action]=(stats.correctionCounts[r.action]??0)+1;
for(const split of ['train','val','test'])stats.splits[split]={detectorImages:detector.filter(r=>r.split===split).length,detectorBoxes:[0,1,2].map(c=>detector.filter(r=>r.split===split).reduce((n,r)=>n+r.annotations.filter(a=>a.classId===c).length,0)),classifierCrops:[0,1].map(c=>crops.filter(r=>r.split===split&&r.classId===c).length),groups:[...allocations.values()].filter(s=>s===split).length};
fs.writeFileSync('reports/preparation/summary.json',JSON.stringify(stats,null,2)+'\n');
fs.writeFileSync(`${out}/training-contract.json`,JSON.stringify({initialization:'random',pretrained:false,externalWeightsAllowed:false,detectorClasses:['moto_con_ocupante','cabeza_con_casco','cabeza_sin_casco'],classifierClasses:['con_casco','sin_casco'],correctFasteningAssessed:false,separateRiderDetectionSupported:false,augmentOnly:'train',evaluation:'Grouped internal baseline; scene independence not certified. External camera/video test still needed.',seed,groupPolicy:'Exact hashes, export lineage, named videos, dHash candidate components and adjacent EdgeVision filename blocks of 25 co-located. Heuristics are not verified scene IDs.'},null,2)+'\n');
console.log(JSON.stringify(stats,null,2));
