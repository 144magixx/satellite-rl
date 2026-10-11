"use strict";
(() => {
  const names = ["MLM掩码布局重构", "强教师与模仿学习BC", "监督学习直接构造布局", "学习候选排序与搜索", "学习区域选择与大邻域重建", "需求约束聚类与归属重分配", "足迹约束聚类与中心投影", "最优传输分区与几何修复", "地理邻接图划分与波束构造", "候选波束池与整数规划", "大邻域搜索与局部整数规划", "多目标进化搜索"];
  const section = document.createElement("section");
  section.id = "page-nondrl"; section.className = "page"; section.hidden = true;
  section.innerHTML = `<div class="page-heading"><div><p class="eyebrow">S1-ND12 · EUROPE</p><h1>非DRL十二路线</h1><p>逐机全覆盖、唯一归属、至多200束。首轮固定0.5°与同束数比较；教师、监督学习、搜索及求解成本分开保存。</p></div><button id="nd-refresh">刷新真实任务</button></div>
  <div class="panel"><h2>路线与冻结配置</h2><div class="field-grid">
  <label class="field">路线<select id="nd-route">${names.map((n,i)=>`<option value="${String(i+1).padStart(2,'0')}">${String(i+1).padStart(2,'0')} ${n}</option>`).join("")}</select></label>
  <label class="field">任务类型<select id="nd-kind"><option value="evaluate">评价</option><option value="prepare">准备</option><option value="train">监督训练</option></select></label>
  <label class="field">准备内容<select id="nd-prep"><option value="initial">共同I0初解</option><option value="teacher">教师标签</option><option value="dataset">官方逐机数据</option></select></label>
  <label class="field">数据用途<select id="nd-role"><option value="development">已见开发</option><option value="validation">新验证</option><option value="test">独立考核</option><option value="train">训练</option></select></label>
  <label class="field">比较方法<select id="nd-method"><option value="main">路线主体</option><option value="I0">共同初解I0</option><option value="BH">共同受限搜索BH</option></select></label>
  <label class="field">任务用途<select id="nd-purpose"><option value="formal">正式实验</option><option value="verification">工程短核验</option></select></label>
  </div><p id="nd-budget" class="small muted"></p><label class="field">严格配置JSON（所有字段经服务端校验）<textarea id="nd-config" rows="14"></textarea></label><button id="nd-validate">校验配置</button></div>
  <div class="panel"><h2>数据与任务来源</h2><p class="small muted">先登记已准备NPZ，或通过准备任务下载官方归档。新日期、UTC抽样及解析器均须预先冻结；任务不能自行换样本。</p>
  <label class="field">数据登记JSON（path、sha256、partition_role、partition_manifest_path及SHA）<textarea id="nd-import-json" rows="5">{}</textarea></label><button id="nd-import">校验并登记NPZ</button>
  <pre id="nd-datasets"></pre><label class="field">任务参数JSON（dataset_ids、共同初解/教师/模型来源、官方日期与解析器）<textarea id="nd-task" rows="10">{"dataset_ids":{},"initial_source":{"port":"20418","run_id":"","route":"06"}}</textarea></label>
  <button id="nd-submit">提交并冻结工作台任务</button><p id="nd-message" class="notice muted"></p></div>
  <div class="panel"><h2>真实任务进度与原始结果</h2><p class="small muted">非学习评价在DQN解释器中使用CPU；神经监督更新使用CUDA。进度来自任务状态，科学停止与执行失败分别记录。</p><div id="nd-runs"></div><pre id="nd-results"></pre></div>`;
  document.getElementById("page-coverage").parentElement.appendChild(section);
  const el = id => document.getElementById("nd-"+id);
  let catalog, loading = false;
  const escape = value => String(value ?? "").replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  async function request(path, body) {
    const response = await fetch("/api/"+path, body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const result = await response.json(); if(!response.ok) throw new Error(result.error||"请求失败"); return result;
  }
  function message(text) { el("message").textContent = text; }
  function defaults() {
    const currentMethod = el("method").value;
    el("method").innerHTML = catalog.methods[el("route").value].map(m=>`<option value="${escape(m)}">${escape(m)}</option>`).join("");
    if(catalog.methods[el("route").value].includes(currentMethod))el("method").value=currentMethod;
    const config = structuredClone(catalog.defaults[el("route").value]);
    config.coverage.nondrl.method = el("method").value;
    el("config").value = JSON.stringify(config,null,2);
    el("budget").textContent = `种子 ${config.training.seed}；监督更新上限 ${config.training.updates}；教师问题 ${config.teacher.max_problems}、试算 ${config.teacher.max_trials}、秒数 ${config.teacher.max_seconds}；在线 ${config.coverage.nondrl.limits.online_seconds}秒，最多 ${config.coverage.nondrl.limits.max_candidates} 候选、${config.coverage.nondrl.limits.max_physical_calls} 全景物理核验。`;
  }
  async function refresh() {
    if(loading) return; loading=true;
    try {
      const [datasets,runs] = await Promise.all([request("coverage/nondrl/datasets"),request("coverage/nondrl/runs")]);
      el("datasets").textContent = JSON.stringify(datasets.datasets.map(d=>({id:d.id,role:d.partition_role,count:d.count,dates:d.dates,sha256:d.sha256})),null,2);
      el("runs").innerHTML = runs.runs.map(r=>`<div class="panel"><strong>${escape(r.name)} · ${escape(r.run_id)}</strong><p>${escape(JSON.stringify({stage:r.stage,kind:r.kind,route:r.variant,device:r.execution_device,completed_scenes:r.completed_scenes,total_scenes:r.total_scenes,updates:r.updates,update_step:r.update_step,budget_updates:r.budget_updates,problems:r.problems,samples:r.samples,trials:r.trials,heartbeat_at:r.heartbeat_at,error:r.error,stop_reason:r.stop_reason,scientific_stop:r.scientific_stop}))}</p><button data-nd-stop="${escape(r.run_id)}">停止</button> <button data-nd-resume="${escape(r.run_id)}">同ID恢复</button> <button data-nd-results="${escape(r.run_id)}">原始结果入口</button></div>`).join("") || "暂无S1-ND12任务";
    } catch(e) {message(e.message);} finally {loading=false;}
  }
  async function guarded(action) {try{await action();}catch(e){message(e.message);}}
  el("route").onchange = ()=>catalog&&defaults(); el("method").onchange=()=>catalog&&defaults();
  el("refresh").onclick=refresh;
  el("validate").onclick=()=>guarded(async()=>{const v=await request("coverage/nondrl/validate",JSON.parse(el("config").value));el("config").value=JSON.stringify(v.config,null,2);message("服务端配置校验通过");});
  el("import").onclick=()=>guarded(async()=>{const r=await request("coverage/nondrl/datasets/import",JSON.parse(el("import-json").value));message("已登记 "+r.id);await refresh();});
  el("submit").onclick=()=>guarded(async()=>{
    const body={...JSON.parse(el("task").value),config:JSON.parse(el("config").value),kind:el("kind").value,experiment_role:el("purpose").value};
    if(body.kind==="prepare")body.preparation_mode=el("prep").value;
    if(body.kind==="evaluate"||body.kind==="prepare"&&["initial","dataset"].includes(body.preparation_mode))body.dataset_role=el("role").value;
    const r=await request("coverage/nondrl/runs",body);message("已冻结并提交 "+r.run_id);await refresh();
  });
  el("runs").onclick=event=>guarded(async()=>{
    const target=event.target;
    if(target.dataset.ndStop) await request("runs/"+target.dataset.ndStop+"/stop",{});
    if(target.dataset.ndResume) await request("runs/"+target.dataset.ndResume+"/resume",{});
    if(target.dataset.ndResults) el("results").textContent=JSON.stringify(await request("runs/"+target.dataset.ndResults+"/results"),null,2);
    await refresh();
  });
  request("coverage/nondrl/catalog").then(value=>{catalog=value;defaults();}).catch(e=>message(e.message));
  setInterval(()=>{if(!section.hidden)refresh();},5000);
})();
