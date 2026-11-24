import test from 'node:test';
import assert from 'node:assert/strict';
import {validateBox,pixelBox,UnionFind,chooseSplits} from './dataset-utils.mjs';
test('rejects invalid boxes rather than silently clipping or labeling background',()=>{
 assert.equal(validateBox([0,0,100,80],100,80),null);
 assert.equal(validateBox([-1,0,100,80],100,80),'outside_image');
 assert.equal(validateBox([10,10,10,50],100,80),'degenerate_box');
 assert.equal(validateBox([0,NaN,10,10],100,80),'invalid_numbers');
 assert.deepEqual(pixelBox({coordinates:[.5,.5,.2,.4]},100,50),[40,15,60,35]);
});
test('transitive relationships prevent splitting derivatives linked through another source',()=>{
 const uf=new UnionFind(['original','crop-source','duplicate','unrelated']);
 uf.join('original','crop-source');uf.join('crop-source','duplicate');
 assert.equal(uf.root('original'),uf.root('duplicate'));
 assert.notEqual(uf.root('original'),uf.root('unrelated'));
 assert.throws(()=>uf.join('missing','original'));
});
test('group allocation is reproducible and independent of input order',()=>{
 const groups=Array.from({length:100},(_,i)=>({id:String(i),size:1,features:{images:1,helmet:i%2,nohelmet:1-i%2}}));
 const a=chooseSplits(groups,'fixed'),b=chooseSplits([...groups].reverse(),'fixed');
 for(const [id,s]of a)assert.equal(b.get(id),s);
 assert.equal([...a.values()].filter(s=>s==='train').length,70);
 assert.equal([...a.values()].filter(s=>s==='val').length,15);
 assert.equal([...a.values()].filter(s=>s==='test').length,15);
});
