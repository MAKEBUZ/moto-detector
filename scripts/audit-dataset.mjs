// Read-only audit of the upstream snapshot. No training or downloads.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';

const root = path.resolve(process.argv[2] ?? 'data/sources/DataSetsIA');
const output = path.resolve(process.argv[3] ?? 'reports/dataset-audit.json');
const base = path.join(root, 'datasets_casco/unificado');
if (!fs.existsSync(base)) throw new Error(`Missing dataset: ${base}`);
function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e =>
    e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]);
}
const relative = f => path.relative(root, f).replaceAll('\\', '/');
const classes = ['motocicleta', 'casco_correcto', 'casco_mal_puesto', 'sin_casco'];
const result = {
  repository: 'https://github.com/Valery-Rosero/DataSetsIA',
  commit: execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  generatedAt: new Date().toISOString(),
  scope: 'Labels, inventory, JPEG dimensions and exact crop hashes; not full image decoding or perceptual deduplication.',
  classes, splits: {}, sources: {}, issues: [], cropDimensions: { unreadableHeaders: [], under12Pixels: 0, under32Pixels: 0 },
};
const hashes = new Map();
const stems = new Map();
const videoGroups = new Map();
function jpegSize(buf) {
  if (buf[0] !== 255 || buf[1] !== 216) return null;
  let i = 2;
  while (i + 8 < buf.length) {
    if (buf[i] !== 255) return null;
    while (buf[i] === 255) i++;
    const marker = buf[i++];
    if (marker === 217 || marker === 218) break;
    if (marker === 1 || (marker >= 208 && marker <= 215)) continue;
    const len = buf.readUInt16BE(i);
    if (len < 2 || i + len > buf.length) return null;
    if ([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker))
      return { height: buf.readUInt16BE(i + 3), width: buf.readUInt16BE(i + 5) };
    i += len;
  }
  return null;
}
for (const split of ['train', 'val', 'test']) {
  const labels = walk(path.join(base, 'labels', split)).filter(f => f.endsWith('.txt'));
  const originals = walk(path.join(base, 'images', split));
  const originalStems = new Set(originals.map(f => path.parse(f).name));
  const stats = result.splits[split] = { labelFiles: labels.length, emptyLabels: 0, originalImages: originals.length,
    labelsWithoutOriginal: 0, boxes: Object.fromEntries(classes.map(c => [c, 0])), crops: {} };
  for (const file of labels) {
    const stem = path.parse(file).name;
    // Evidence from filenames only; not proof of original video identity.
    const video = stem.match(/^(.*?VID_\d{8}_\d{6}_mp4)-/);
    if (video) {
      const group = videoGroups.get(video[1]) ?? {};
      group[split] = (group[split] ?? 0) + 1;
      videoGroups.set(video[1], group);
    }
    const source = stem.split('__')[0];
    const src = result.sources[source] ??= { labelFiles: 0, boxes: Object.fromEntries(classes.map(c => [c, 0])) };
    src.labelFiles++;
    if (!originalStems.has(stem)) stats.labelsWithoutOriginal++;
    const groups = stems.get(stem) ?? []; groups.push(split); stems.set(stem, groups);
    const content = fs.readFileSync(file, 'utf8').trim();
    if (!content) { stats.emptyLabels++; continue; }
    const seen = new Set();
    content.split(/\r?\n/).forEach((line, index) => {
      const vals = line.trim().split(/\s+/).map(Number);
      const issue = reason => result.issues.push({ file: relative(file), line: index + 1, reason });
      if (vals.length !== 5 || !vals.every(Number.isFinite)) { issue('invalid_format'); return; }
      const [c, x, y, w, h] = vals;
      if (!Number.isInteger(c) || c < 0 || c >= classes.length) { issue('invalid_class'); return; }
      stats.boxes[classes[c]]++; src.boxes[classes[c]]++;
      if (w <= 0 || h <= 0 || vals.slice(1).some(v => v < 0 || v > 1)) issue('invalid_normalized_coordinates');
      if (x-w/2 < -0.00001 || y-h/2 < -0.00001 || x+w/2 > 1.00001 || y+h/2 > 1.00001) issue('box_outside_image');
      const key = vals.join(' ');
      if (seen.has(key)) issue('duplicate_box'); seen.add(key);
    });
  }
  for (const cls of classes.slice(1)) {
    const files = walk(path.join(base, 'recortes', split, cls));
    stats.crops[cls] = files.length;
    for (const file of files) {
      const buf = fs.readFileSync(file);
      const hash = crypto.createHash('sha256').update(buf).digest('hex');
      const group = hashes.get(hash) ?? []; group.push({ file: relative(file), split, class: cls }); hashes.set(hash, group);
      const size = jpegSize(buf);
      if (!size) result.cropDimensions.unreadableHeaders.push(relative(file));
      else {
        if (Math.min(size.width, size.height) < 12) result.cropDimensions.under12Pixels++;
        if (Math.min(size.width, size.height) < 32) result.cropDimensions.under32Pixels++;
      }
    }
  }
}
result.repeatedLabelStemsAcrossSplits = [...stems].filter(([,v]) => new Set(v).size > 1);
result.inferredVideoGroups = Object.fromEntries(videoGroups);
result.inferredCrossSplitVideoGroups = Object.fromEntries([...videoGroups].filter(([,v]) => Object.keys(v).length > 1));
result.exactDuplicateCropGroups = [...hashes.values()].filter(v => v.length > 1);
result.crossSplitDuplicateCropGroups = result.exactDuplicateCropGroups.filter(v => new Set(v.map(x => x.split)).size > 1);
result.conflictingClassCropGroups = result.exactDuplicateCropGroups.filter(v => new Set(v.map(x => x.class)).size > 1);
result.issueCounts = result.issues.reduce((a, x) => { a[x.reason] = (a[x.reason] ?? 0) + 1; return a; }, {});
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ commit: result.commit, splits: result.splits, sources: result.sources,
  issueCounts: result.issueCounts, cropDimensions: result.cropDimensions,
  exactDuplicateGroups: result.exactDuplicateCropGroups.length,
  crossSplitDuplicateGroups: result.crossSplitDuplicateCropGroups.length,
  conflictingClassGroups: result.conflictingClassCropGroups.length,
  repeatedLabelStems: result.repeatedLabelStemsAcrossSplits.length, output }, null, 2));
