#!/usr/bin/env node
/** GET-only Graph export. Account-specific policy and authentication live outside this repo. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
const hash = s => crypto.createHash('sha256').update(s).digest('hex');
export function graphUrl(url) {
  const u = new URL(url);
  if (u.origin !== 'https://graph.microsoft.com' || !u.pathname.startsWith('/v1.0/me/')) throw Error('Untrusted Graph continuation');
  return u.href;
}
export function cleanBody(text) {
  // Conservative quote removal: only explicit quote lines and standard reply separators.
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const stop = lines.findIndex(l => /^\s*(?:On .+wrote:|Am .+schrieb .+:|-----Original Message-----|_{5,})\s*$/i.test(l));
  return (stop < 0 ? lines : lines.slice(0, stop)).filter(l => !/^\s*>/.test(l)).join('\n').trim();
}
const excluded = p => /^(Archive|Deleted Items)(\/|$)/i.test(p);
const filterHeaders = hs => (hs || []).filter(h=>['list-unsubscribe','list-id','precedence','auto-submitted'].includes(h.name.toLowerCase())).map(h=>({name:h.name,value:['list-unsubscribe','list-id'].includes(h.name.toLowerCase())?'present':h.value}));
const oneLine = s => String(s || '').replace(/[\r\n]/g, ' ');
export async function sync({root, get, policy}) {
  if (!policy.version || !policy.select) throw Error('Versioned external policy required');
  fs.mkdirSync(root, {recursive:true, mode:0o700});
  fs.chmodSync(root, 0o700);
  const lock = path.join(root, '.sync.lock');
  const fd = fs.openSync(lock, 'wx', 0o600);
  let staging;
  try {
    const current = path.join(root, 'current');
    const prior = fs.existsSync(current) ? JSON.parse(fs.readFileSync(path.join(current, 'state.json'))) : {folders:{}, records:{}, policy:null};
    const folders = [];
    async function walk(id='', prefix='') {
      let url = `https://graph.microsoft.com/v1.0/me/mailFolders${id ? `/${encodeURIComponent(id)}/childFolders` : ''}?$top=100&includeHiddenFolders=true&$select=id,displayName,childFolderCount,totalItemCount`;
      while (url) {
        const page = await get(graphUrl(url));
        for (const f of page.value) {
          const name = prefix + f.displayName;
          if (excluded(name)) continue;
          folders.push({id:f.id, name, total:f.totalItemCount || 0});
          if (f.childFolderCount) await walk(f.id, name+'/');
        }
        url = page['@odata.nextLink'];
      }
    }
    await walk();
    // Full rescan on policy change: excluded messages intentionally retain no body in state.
    const reset = prior.policy !== policy.version;
    const records = reset ? {} : {...prior.records};
    const cursors = {};
    const folderNames = new Map(folders.map(f=>[f.id,f.name]));
    const folderIds = new Set(folders.map(f=>f.id));
    for (const [id,m] of Object.entries(records)) {
      if (!folderIds.has(m.folderId)) delete records[id];
      else m.folder = folderNames.get(m.folderId);
    }
    for (const f of folders) {
      let url = !reset && prior.folders[f.id] || `https://graph.microsoft.com/v1.0/me/mailFolders/${encodeURIComponent(f.id)}/messages/delta?$select=id,subject,from,toRecipients,ccRecipients,bccRecipients,receivedDateTime,lastModifiedDateTime,conversationId,internetMessageId,internetMessageHeaders,hasAttachments,body`;
      const visited = new Set();
      while (url) {
        if (visited.has(url)) throw Error('Repeated Graph continuation');
        visited.add(url);
        const page = await get(graphUrl(url));
        for (const m of page.value) {
          if (m['@removed']) {
            if (records[m.id]?.folderId === f.id) delete records[m.id];
          } else {
            if (m.internetMessageHeaders) m.internetMessageHeaders = filterHeaders(m.internetMessageHeaders);
            records[m.id] = {...records[m.id], ...m, folder:f.name, folderId:f.id};
          }
        }
        url = page['@odata.nextLink'];
        if (!url) {
          if (!page['@odata.deltaLink']) throw Error('Incomplete delta response');
          cursors[f.id] = graphUrl(page['@odata.deltaLink']);
        }
      }
    }
    // Policy receives whole current metadata snapshot for stable cross-message identity aliases.
    const selected = policy.select(Object.values(records));
    if (!(selected instanceof Set)) throw Error('Policy must return Set of selected ids');
    let fetched = 0;
    for (const id of selected) {
      const m = records[id];
      if (!m || excluded(m.folder)) throw Error('Policy selected an invalid record');
      if (!m.body) {
        const detail = await get(graphUrl(`https://graph.microsoft.com/v1.0/me/messages/${encodeURIComponent(id)}?$select=body`));
        m.body = detail.body;
        fetched++;
      }
      if (!m.body || m.body.contentType.toLowerCase() !== 'text') throw Error('Graph did not return text body');
    }
    staging = fs.mkdtempSync(path.join(root, '.generation-'));
    const mailDir = path.join(staging, 'mail');
    fs.mkdirSync(mailDir, {mode:0o700});
    const seen = new Set();
    const aliases = {};
    let bytes=0, duplicates=0, unchanged=0;
    for (const id of [...selected].sort()) {
      const m = records[id];
      const body = /^(?:re|aw|antw)\s*:/i.test(m.subject || '') ? cleanBody(m.body.content) : m.body.content.trim();
      // Do not collapse a conversation: distinct replies carry distinct facts.
      const contentKey = hash(JSON.stringify([oneLine(m.subject).replace(/^(?:(?:re|aw|fw|fwd|wg):\s*)+/ig,''), m.from, m.receivedDateTime, body]));
      const key = hash(m.internetMessageId ? 'message:'+m.internetMessageId : 'content:'+contentKey);
      const file = key+'.md';
      aliases[id] = file;
      if (seen.has(key)) {duplicates++; continue;}
      seen.add(key);
      const text = `# ${oneLine(m.subject)}\n\nSource: mail\nFrom: ${oneLine(m.from?.emailAddress?.name)} <${oneLine(m.from?.emailAddress?.address)}>\nDate: ${oneLine(m.receivedDateTime)}\n\n${body}\n`;
      bytes += Buffer.byteLength(text);
      const old = path.join(current, 'mail', file);
      if (fs.existsSync(old) && fs.readFileSync(old,'utf8')===text) unchanged++;
      fs.writeFileSync(path.join(mailDir,file), text, {mode:0o600});
      // Original selected body remains private sync state for repeatable body-prefix policy.
    }
    // Metadata is private state, not indexed. Excluded bodies are never persisted.
    for (const m of Object.values(records)) {
      if (!selected.has(m.id)) delete m.body;
      m.internetMessageHeaders = filterHeaders(m.internetMessageHeaders);
    }
    const state = {version:1, policy:policy.version, folders:cursors, records, aliases};
    fs.writeFileSync(path.join(staging,'state.json'),JSON.stringify(state),{mode:0o600});
    const stats = {folderItems:folders.reduce((n,f)=>n+f.total,0), crawled:Object.keys(records).length, included:selected.size, exported:seen.size, duplicates, unchanged, bytes, fetched};
    fs.writeFileSync(path.join(staging,'stats.json'),JSON.stringify(stats),{mode:0o600});
    const link = path.join(root,'.next');
    fs.symlinkSync(path.basename(staging),link);
    fs.renameSync(link,current);
    // Keep no stale generations containing removed or newly excluded messages.
    for (const name of fs.readdirSync(root)) if (name.startsWith('.generation-') && name!==path.basename(staging)) fs.rmSync(path.join(root,name),{recursive:true});
    return stats;
  } finally {
    fs.closeSync(fd); fs.unlinkSync(lock);
    if (staging && (!fs.existsSync(path.join(root,'current')) || fs.realpathSync(path.join(root,'current'))!==staging)) fs.rmSync(staging,{recursive:true,force:true});
  }
}
if (process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  // Adapter exports get(url), with token refresh/retry if needed, and policy.
  const [adapter, root] = process.argv.slice(2);
  if (!adapter || !root) throw Error('Usage: export_mail.mjs /private/adapter.mjs /private/export-root');
  const {get,policy} = await import(pathToFileURL(path.resolve(adapter)));
  console.log(JSON.stringify(await sync({root:path.resolve(root), get, policy})));
}
