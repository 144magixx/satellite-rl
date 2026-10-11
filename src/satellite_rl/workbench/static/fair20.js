"use strict";
const element = id => document.getElementById(id);
let schemaCache, taskCache=[];
async function api(path, body) {
  const response = await fetch('/api/fair20/' + path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || '请求失败');
  return value;
}
function show(id, value) { element(id).textContent = typeof value === 'string' ? value : JSON.stringify(value,null,2); }
function populate(id, rows) {
  const select=element(id), previous=select.value;
  select.replaceChildren();
  const empty=document.createElement('option');empty.value='';empty.textContent='选择已完成来源';select.append(empty);
  for(const {job} of rows){const option=document.createElement('option');option.value=job.run_id;option.textContent=job.run_id;select.append(option);}
  if(rows.some(({job})=>job.run_id===previous))select.value=previous;
}
function updateSources() {
  const completed=taskCache.filter(({status})=>status.stage==='completed');
  const algorithm=element('algorithm').value;
  populate('prepare-source',completed.filter(({job})=>element('mode').value==='prepare'?job.mode==='sourceprep':['prepare','sourceprep'].includes(job.mode)));
  populate('tune-a-source',completed.filter(({job})=>job.mode==='shorttune'&&job.algorithm===algorithm&&job.candidate==='A'));
  populate('tune-b-source',completed.filter(({job})=>job.mode==='shorttune'&&job.algorithm===algorithm&&job.candidate==='B'));
  populate('model-source',completed.filter(({job})=>job.mode==='train'&&job.algorithm===algorithm));
  populate('direct-source',completed.filter(({job})=>job.mode==='evaluate'));
  for(const method of ['sac','ppo','dqn'])populate('matrix-'+method+'-source',completed.filter(({job})=>job.mode==='train'&&job.algorithm===method));
}
async function updateConfig() {
  const mode=element('mode').value;
  const prepare=['sourceprep','prepare','matrix'].includes(mode);
  const algorithm=prepare?'prepare':mode==='refine'?'sac':element('algorithm').value;
  element('prepare-field').hidden=!['prepare','shorttune','train','evaluate','matrix'].includes(mode);
  element('matrix-model-fields').hidden=mode!=='matrix';
  element('tune-a-field').hidden=mode!=='train';element('tune-b-field').hidden=mode!=='train';
  element('model-field').hidden=mode!=='evaluate'||['random','greedy'].includes(algorithm);
  element('direct-field').hidden=mode!=='refine';element('candidate').disabled=mode!=='shorttune';
  element('algorithm').disabled=prepare||mode==='refine';
  if(prepare||mode==='refine')element('candidate').value='A';
  if(mode==='train'&&element('tune-a-source').value&&element('tune-b-source').value){
    try{
      const [a,b]=await Promise.all(['tune-a-source','tune-b-source'].map(async id=>(await api('tasks/'+element(id).value)).result.validation_summary));
      element('candidate').value=Math.abs(a.SI-b.SI)>.001?(b.SI>a.SI?'B':'A'):(b.SGM>a.SGM?'B':'A');
    }catch(error){show('response',error.message);}
  }
  if(mode==='evaluate'&&element('model-source').value){const source=taskCache.find(({job})=>job.run_id===element('model-source').value);if(source)element('candidate').value=source.job.candidate;}
  const key=`${mode}/${algorithm}/${element('candidate').value}`;
  show('schema',schemaCache?.configurations[key]||'请选择与任务相符的算法。');
}
async function refresh() {
  try {
    schemaCache = await api('schema');
    const value = await api('tasks');
    taskCache=value.tasks;updateSources();await updateConfig();
    element('tasks').replaceChildren();
    for (const {job,status} of value.tasks) {
      const row = document.createElement('div'); row.className='notice';
      const label = document.createElement('span');
      label.textContent = `${job.run_id} · ${job.mode} · ${job.algorithm} · ${status.stage} · ${status.env_step||0}/${status.budget_env_steps||0} · ${status.completed_scenes||0}景 `;
      if(job.execution_source)label.textContent+=`科学源 ${job.git_commit.slice(0,8)} · 修复执行源 ${job.execution_source.execution_commit.slice(0,8)} `;
      row.append(label);
      for (const [caption,action] of [['详情与日志','detail'],['停止','stop'],['同ID恢复','resume']]) {
        const button=document.createElement('button'); button.type='button'; button.textContent=caption;
        button.addEventListener('click',async()=>{
          try {show('detail',await api('tasks/'+encodeURIComponent(job.run_id)+(action==='detail'?'':'/'+action),action==='detail'?undefined:{}));await refresh();}
          catch(error){show('detail',error.message);}
        });row.append(button);
      }
      element('tasks').append(row);
    }
    if(!value.tasks.length)element('tasks').textContent='尚无S2-F20任务。';
  } catch(error) { show('response',error.message); }
}
element('submit').addEventListener('click',async()=>{
  const button=element('submit');button.disabled=true;
  try {
    const body=JSON.parse(element('payload').value);const mode=element('mode').value;
    body.request_id=element('request-id').value.trim();
    if(!body.request_id)throw new Error('请填写唯一请求编号');
    if(!['sourceprep','prepare','refine','matrix'].includes(mode))body.algorithm=element('algorithm').value;
    if(mode==='shorttune')body.candidate=element('candidate').value;
    if(['shorttune','train','evaluate','matrix'].includes(mode)&&element('prepare-source').value)body.prepare_run_id=element('prepare-source').value;
    if(mode==='prepare'&&element('prepare-source').value)body.sourceprep_run_id=element('prepare-source').value;
    if(mode==='train'&&element('tune-a-source').value&&element('tune-b-source').value)body.selection_run_ids=[element('tune-a-source').value,element('tune-b-source').value];
    if(mode==='evaluate'&&element('model-source').value)body.model_run_id=element('model-source').value;
    if(mode==='refine'&&element('direct-source').value)body.direct_run_id=element('direct-source').value;
    if(mode==='matrix'){const models={};for(const method of ['sac','ppo','dqn'])if(element('matrix-'+method+'-source').value)models[method]=element('matrix-'+method+'-source').value;if(Object.keys(models).length===3)body.models_by_algorithm=models;}
    show('response',await api(mode,body));await refresh();
  }catch(error){show('response',error.message);}finally{button.disabled=false;}
});
element('refresh').addEventListener('click',refresh);
for(const id of ['mode','algorithm','candidate','prepare-source','tune-a-source','tune-b-source','model-source','direct-source','matrix-sac-source','matrix-ppo-source','matrix-dqn-source'])element(id).addEventListener('change',async()=>{if(['algorithm','mode'].includes(id))updateSources();await updateConfig();});
refresh();
