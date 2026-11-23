import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {sync,cleanBody,graphUrl} from '../scripts/export_mail.mjs';
const record=(id, extra={})=>({id,subject:'Roof invoice',from:{emailAddress:{name:'Example',address:'person@example.invalid'}},receivedDateTime:'2026-01-02',body:{contentType:'text',content:'Seven-year warranty\n\nOn Tuesday Someone wrote:\nOld facts'},...extra});
const policy={version:'invented-v1',select:records=>new Set(records.filter(m=>m.subject==='Roof invoice').map(m=>m.id))};
function graph(changes, cursor='https://graph.microsoft.com/v1.0/me/delta?cursor=1') {
 return async url=>url.includes('mailFolders?')?{value:[{id:'inbox',displayName:'Inbox'},{id:'archive',displayName:'Archive'},{id:'deleted',displayName:'Deleted Items'}]}:{value:changes,'@odata.deltaLink':cursor};
}
test('quotes and URL validation',()=>{
 assert.equal(cleanBody('Answer\n> old\nAm Dienstag schrieb Example:\nold'),'Answer');
 assert.throws(()=>graphUrl('https://evil.invalid/v1.0/me/messages'));
 assert.throws(()=>graphUrl('https://graph.microsoft.com/v1.0/users/other'));
});
test('snapshot incremental sync, duplicates, deletion, policy exclusion and rollback',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'mail-fixture-'));
 try {
  const first=await sync({root,get:graph([record('one',{internetMessageId:'<copy@example.invalid>'}),record('two',{internetMessageId:'<copy@example.invalid>'}),record('noise',{subject:'Unwanted',body:{contentType:'text',content:'secret excluded'}})]),policy});
  assert.equal(first.exported,1);assert.equal(first.duplicates,1);
  const state=JSON.parse(fs.readFileSync(path.join(root,'current/state.json')));
  assert.equal(state.records.noise.body,undefined);
  assert.equal(fs.readdirSync(path.join(root,'current/mail')).length,1);
  const second=await sync({root,get:graph([]),policy});
  assert.equal(second.unchanged,1);
  const before=fs.readlinkSync(path.join(root,'current'));
  await assert.rejects(sync({root,get:async()=>{throw Error('Network down');},policy}));
  assert.equal(fs.readlinkSync(path.join(root,'current')),before);
  const third=await sync({root,get:graph([{id:'one','@removed':{}},record('two',{subject:'Unwanted'})]),policy});
  assert.equal(third.exported,0);
  assert.equal(fs.readdirSync(root).filter(n=>n.startsWith('.generation-')).length,1);
 } finally {fs.rmSync(root,{recursive:true});}
});
test('rejects HTML responses and incomplete deltas without publishing',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'mail-fixture-'));
 try {
  await assert.rejects(sync({root,get:graph([record('one',{body:{contentType:'html',content:'<b>not text</b>'}})]),policy}),/text body/);
  assert.equal(fs.existsSync(path.join(root,'current')),false);
 } finally {fs.rmSync(root,{recursive:true});}
});
test('all delta pages, no total-change top cap, denied folders and quote-aware forwarding',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'mail-fixture-'));
 const calls=[];
 const get=async url=>{
  calls.push(url);
  if(url.includes('mailFolders?'))return {value:[{id:'inbox',displayName:'Inbox'},{id:'denied',displayName:'Archive',childFolderCount:1}]};
  assert.ok(!url.includes('$top='));
  assert.ok(!url.includes('denied'));
  if(url.includes('page=2'))return {value:[record('reply',{subject:'Roof invoice',internetMessageId:'<second@example.invalid>'})],'@odata.deltaLink':'https://graph.microsoft.com/v1.0/me/delta?done=1'};
  return {value:[record('forward',{subject:'Roof invoice',body:{contentType:'text',content:'Forwarded letter\n-----Original Message-----\nImportant original fact'}})],'@odata.nextLink':'https://graph.microsoft.com/v1.0/me/delta?page=2'};
 };
 try{
  const stats=await sync({root,get,policy});assert.equal(stats.exported,2);assert.equal(calls.length,3);
  const text=fs.readdirSync(path.join(root,'current/mail')).map(f=>fs.readFileSync(path.join(root,'current/mail',f),'utf8')).join('\n');
  assert.match(text,/Important original fact/);
 }finally{fs.rmSync(root,{recursive:true});}
});
test('shared exclusive lock and repeated continuation fail closed',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'mail-fixture-'));
 try{
  fs.writeFileSync(path.join(root,'.sync.lock'),'');
  await assert.rejects(sync({root,get:graph([]),policy}),/EEXIST/);
  fs.unlinkSync(path.join(root,'.sync.lock'));
  const get=async url=>url.includes('mailFolders?')?{value:[{id:'inbox',displayName:'Inbox'}]}:{value:[],'@odata.nextLink':'https://graph.microsoft.com/v1.0/me/delta?repeat=1'};
  await assert.rejects(sync({root,get,policy}),/Repeated Graph continuation/);
 }finally{fs.rmSync(root,{recursive:true});}
});
test('move tombstone cannot delete a message already seen in its new folder',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'mail-fixture-'));
 let incremental=false;
 const get=async url=>{
  if(url.includes('mailFolders?'))return {value:[{id:'new',displayName:'New folder'},{id:'old',displayName:'Old folder'}]};
  const isNew=url.includes('/new/')||url.includes('cursor=new');
  return {value:incremental?(isNew?[record('moved')]:[{id:'moved','@removed':{}}]):(isNew?[]:[record('moved')]),'@odata.deltaLink':`https://graph.microsoft.com/v1.0/me/delta?cursor=${isNew?'new':'old'}`};
 };
 try{
  await sync({root,get,policy});incremental=true;
  const result=await sync({root,get,policy});assert.equal(result.exported,1);
  const state=JSON.parse(fs.readFileSync(path.join(root,'current/state.json')));
  assert.equal(state.records.moved.folder,'New folder');
 }finally{fs.rmSync(root,{recursive:true});}
});
