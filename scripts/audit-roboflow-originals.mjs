import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const lock=JSON.parse(fs.readFileSync('data/sources.lock.json','utf8'));
const mappings={
 motorcycle_helmet_od:{'With Helmet':1,'Without Helmet':3},
 mapua_helmet:{rider_full_face:1,rider_half_face:1,rider_helmet_invalid:2,rider_no_helmet:3,rider:null},
 ldrago_helmet:{'Helmet-Valid':1,'Helmet-on-hand':2,Scarf:2,Turban:2,'No-Helmet-on-Rider':3,'No-Helmet-on-passenger':3,shirt:null},
 proper_helmet_usage:{Proper:1,Improper:2},
 helmet_violation:{helmet:1,cap:2,no_helmet:3,motorcycle:0,person:null}
};
const summary={generatedAt:new Date().toISOString(),sources:{},issues:[],exactDuplicateGroups:[],crossSplitDuplicateGroups:[],inferredSharedVideoGroups:{}};
const hashes=new Map(),videos=new Map();
const records=[];
const rel=f=>f.replaceAll('\\','/');
for(const source of lock.sources.filter(s=>s.provider==='roboflow')){
 const base=`data/sources/originals/${source.id}-v${source.version}`;
 const yaml=fs.readFileSync(`${base}/data.yaml`,'utf8');
 const match=yaml.match(/^names:\s*\[(.*?)\]/m);
 if(!match)throw Error(`Unsupported names format for ${source.id}; manual review required`);
 const names=match[1].split(',').map(x=>x.trim().replace(/^['"]|['"]$/g,''));
 const stats=summary.sources[source.id]={version:source.version,names,images:0,labels:0,missingLabels:0,emptyLabels:0,originalBoxes:0,discardedBoxes:0,matchingUnifiedLabels:0,unifiedMismatches:0,splits:{}};
 for(const [folder,split] of [['train','train'],['valid','val'],['test','test']]){
  const dir=`${base}/${folder}/images`;
  if(!fs.existsSync(dir))continue;
  const imgs=fs.readdirSync(dir).filter(f=>/\.(jpg|jpeg|png)$/i.test(f)).sort();
  const labelsDir=`${base}/${folder}/labels`;
  const labelNames=fs.existsSync(labelsDir)?fs.readdirSync(labelsDir).filter(f=>f.endsWith('.txt')):[];
  stats.labels+=labelNames.length;
  stats.splits[split]=imgs.length;
  const imageStems=new Set(imgs.map(f=>path.parse(f).name));
  for(const l of labelNames)if(!imageStems.has(path.parse(l).name))summary.issues.push({type:'label_without_image',file:`${labelsDir}/${l}`});
  for(const image of imgs){
   stats.images++;
   const stem=path.parse(image).name,file=`${dir}/${image}`,label=`${labelsDir}/${stem}.txt`;
   const data=fs.readFileSync(file),sha256=crypto.createHash('sha256').update(data).digest('hex');
   const group=hashes.get(sha256)??[];group.push({file,source:source.id,split});hashes.set(sha256,group);
   const video=stem.match(/^(.*?VID_\d{8}_\d{6}_mp4)-/);
   const groupId=video?`${source.id}/${video[1]}`:null;
   if(groupId){const v=videos.get(groupId)??{};v[split]=(v[split]??0)+1;videos.set(groupId,v);}
   const record={source:source.id,version:source.version,image:rel(file),sha256,upstreamSplit:split,groupId,groupStatus:groupId?'inferred_from_filename':'scene_unknown',originalLabel:fs.existsSync(label)?rel(label):null,annotations:[],unifiedLabelsMatch:false};
   const expected=[];
   if(!record.originalLabel){stats.missingLabels++;summary.issues.push({type:'missing_label',file});}
   else {
    const text=fs.readFileSync(label,'utf8').trim();if(!text)stats.emptyLabels++;
    const lines=text?text.split(/\r?\n/):[];
    for(const [i,line] of lines.entries()){
     const v=line.trim().split(/\s+/).map(Number),id=v[0],coords=v.slice(1),name=names[id];
     if(!v.every(Number.isFinite)||!Number.isInteger(id)||!name||!(coords.length===4||(coords.length>=6&&coords.length%2===0))){summary.issues.push({type:'invalid_original_label',file:rel(label),line:i+1});continue;}
     stats.originalBoxes++;record.annotations.push({class:name,coordinates:coords});
     if(coords.some(x=>x<0||x>1))summary.issues.push({type:'coordinate_outside_range',file:rel(label),line:i+1});
     let box=coords;
     if(coords.length!==4){const xs=coords.filter((_,i)=>i%2===0),ys=coords.filter((_,i)=>i%2===1);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);box=[(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0];}
     const mapped=mappings[source.id][name];
     if(mapped===undefined){summary.issues.push({type:'unknown_mapping',file:rel(label),class:name});continue;}
     if(mapped===null){stats.discardedBoxes++;continue;}
     if(box[2]>0&&box[3]>0)expected.push([mapped,...box]);
    }
   }
   const unified=`data/sources/DataSetsIA/datasets_casco/unificado/labels/${split}/${source.id}__${stem}.txt`;
   if(fs.existsSync(unified)){
    const text=fs.readFileSync(unified,'utf8').trim();const actual=text?text.split(/\r?\n/).map(l=>l.split(/\s+/).map(Number)):[];
    record.unifiedLabelsMatch=actual.length===expected.length&&actual.every((v,i)=>v.length===5&&v.every((n,j)=>Math.abs(n-expected[i][j])<=0.000002));
   }
   if(record.unifiedLabelsMatch)stats.matchingUnifiedLabels++;else{stats.unifiedMismatches++;summary.issues.push({type:'unified_mismatch_or_missing',file:rel(file),expectedUnified:unified});}
   records.push(record);
  }
 }
 console.log(`${source.id}: ${stats.images} images; ${stats.matchingUnifiedLabels} unified matches`);
}
summary.exactDuplicateGroups=[...hashes.values()].filter(g=>g.length>1);
summary.crossSplitDuplicateGroups=summary.exactDuplicateGroups.filter(g=>new Set(g.map(x=>x.split)).size>1);
summary.inferredSharedVideoGroups=Object.fromEntries([...videos].filter(([,g])=>Object.keys(g).length>1));
summary.issueCounts=summary.issues.reduce((a,x)=>(a[x.type]=(a[x.type]??0)+1,a),{});
fs.writeFileSync('reports/roboflow-manifest.jsonl',records.map(r=>JSON.stringify(r)).join('\n')+'\n');
fs.writeFileSync('reports/roboflow-originals-audit.json',JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify({issueCounts:summary.issueCounts,exactDuplicateGroups:summary.exactDuplicateGroups.length,crossSplitDuplicateGroups:summary.crossSplitDuplicateGroups.length}));
