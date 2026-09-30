"use strict";
(() => {
  let config, datasets = [], models = [], runId = "", loading = false;
  const el = (id) => document.getElementById("coverage-" + id);
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (x) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
  async function request(path, body) {
    const response = await fetch("/api/" + path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "请求失败"); return result;
  }
  function message(value, failed = false) { el("message").textContent = value; el("message").className = "notice " + (failed ? "bad" : "muted"); }
  const coverageFields = {"initialization":"initialization", "method":"method", "sweeps":"sweeps", "coordinate-step":"coordinate_step_deg", "max-movement":"max_movement_deg", "demand-floor":"min_coverage_demand", "user-floor":"min_coverage_users", "cluster-iterations":"cluster_iterations"};
  const trainingFields = {"seed":"seed", "total-steps":"total_steps", "num-envs":"num_envs", "rollout-steps":"rollout_steps", "minibatch":"minibatch_size", "epochs":"epochs", "lr":"lr", "validation-interval":"validation_interval", "checkpoint-interval":"checkpoint_interval"};
  function writeForm() {
    el("name").value = config.name;
    for (const [id,key] of Object.entries(coverageFields)) el(id).value = config.coverage[key];
    for (const [id,key] of Object.entries(trainingFields)) el(id).value = config.training[key];
    el("config-json").value = JSON.stringify(config,null,2); renderDatasets();
  }
  function readForm() {
    config.name = el("name").value;
    for (const [id,key] of Object.entries(coverageFields)) config.coverage[key] = ["initialization","method"].includes(key) ? el(id).value : Number(el(id).value);
    for (const [id,key] of Object.entries(trainingFields)) config.training[key] = Number(el(id).value);
    el("config-json").value = JSON.stringify(config,null,2); return config;
  }
  function renderDatasets() {
    for (const role of ["train","validation","test"]) {
      const node = el("data-" + role), previous = node.value;
      node.innerHTML = '<option value="">请选择已准备数据</option>' + datasets.filter((r) => r.partition_role === role && r.initialization === config.coverage.initialization).map((r) => `<option value="${escape(r.id)}">${escape(r.name)} · ${r.count}场景 · ${escape((r.dates||[]).join(","))}</option>`).join("");
      if ([...node.options].some((r) => r.value === previous)) node.value = previous;
    }
  }
  function curve(series) {
    const points = series.points || []; if (points.length < 2) return "<p>尚无可绘制曲线</p>";
    const xs = points.map((p) => Number(p.step)), ys = points.map((p) => Number(p.value));
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const path = points.map((p,i) => `${i?"L":"M"}${16+268*(p.step-xmin)/(xmax-xmin||1)},${102-86*(p.value-ymin)/(ymax-ymin||1)}`).join(" ");
    return `<svg viewBox="0 0 300 120" role="img" aria-label="${escape(series.tag)}原始曲线"><path d="${path}" fill="none" stroke="#2563eb" stroke-width="1.8"/><text x="16" y="117" font-size="9">${xmin}–${xmax}步 · ${ymin.toFixed(4)}–${ymax.toFixed(4)}</text></svg>`;
  }
  async function runDetail() {
    if (!runId || loading) return; loading = true;
    try {
      const [detail,logs,results,metrics] = await Promise.all([request("runs/"+encodeURIComponent(runId)),request("runs/"+encodeURIComponent(runId)+"/logs"),request("runs/"+encodeURIComponent(runId)+"/results"),request("runs/"+encodeURIComponent(runId)+"/metrics?limit=300")]);
      el("status").innerHTML = `<p>状态：${escape(detail.status.stage)} · ${detail.status.env_step||0}/${detail.status.budget_env_steps||0}交互步 · ${detail.status.update_step||0}次更新</p><p>考核：${detail.status.evaluated_scenes||0}/${detail.status.evaluation_total||0}场景</p><p>源码：${escape(detail.job.git_commit)}<br>GPU：${escape(detail.status.preflight?.gpu||detail.status.gpu||"")}<br>冻结配置与来源保留于任务记录。</p><details><summary>任务配置与状态</summary><pre>${escape(JSON.stringify(detail,null,2))}</pre></details>`;
      el("logs").textContent = (logs.lines||[]).join("\n");
      el("results").textContent = JSON.stringify(results.summary||results,null,2);
      const list = (metrics.series||metrics.tags||[]); el("curves").innerHTML = list.slice(0,10).map((s) => `<div><p>${escape(s.tag||s.name)}</p>${curve(s)}</div>`).join("") || "暂无原始曲线";
    } catch (error) {message(error.message,true);} finally {loading = false;}
  }
  async function refresh() {
    const [data, modelData, runs] = await Promise.all([request("coverage/datasets"),request("coverage/models"),request("coverage/runs")]);
    datasets=data.datasets;models=modelData.models;renderDatasets();
    const prior=el("model").value;el("model").innerHTML='<option value="">选择第一阶段模型</option>'+models.map((m)=>`<option value="${escape(m.id)}">${escape(m.name)}</option>`).join("");if(models.some((m)=>m.id===prior))el("model").value=prior;
    el("run").innerHTML='<option value="">选择任务</option>'+runs.runs.map((r)=>`<option value="${escape(r.run_id)}">${escape(r.name)} · ${escape(r.stage)}</option>`).join("");if(runs.runs.some((r)=>r.run_id===runId))el("run").value=runId;await runDetail();
  }
  async function guarded(fn) {try {await fn();}catch(error){message(error.message,true);}}
  document.addEventListener("DOMContentLoaded", async () => {
    try {config=(await request("coverage/catalog")).defaults;writeForm();await refresh();message("第一阶段配置已加载；正式任务在当前连接的远端工作台执行。");}catch(error){message(error.message,true);return;}
    el("refresh").onclick=()=>guarded(refresh);
    el("initialization").onchange=()=>{readForm();renderDatasets();};
    el("validate").onclick=()=>guarded(async()=>{const result=await request("coverage/validate",readForm());config=result.config;writeForm();message(`校验通过：${result.summary.state_dim}维状态，${result.summary.action_dim}个中心动作；评分差分，gamma=1。`);});
    el("apply-json").onclick=()=>guarded(async()=>{const result=await request("coverage/validate",JSON.parse(el("config-json").value));config=result.config;writeForm();message("完整配置已校验并应用。");});
    el("save").onclick=()=>guarded(async()=>{const saved=await request("coverage/configs",{config:readForm()});message("方案已保存："+saved.id);});
    el("import-data").onclick=()=>guarded(async()=>{message("正在按基础布局准备逐机数据…");const record=await request("coverage/datasets/import",{path:el("data-path").value,partition_role:el("data-role").value,coverage:readForm().coverage});await refresh();el("data-"+record.partition_role).value=record.id;message(`已保存${record.count}个场景；初始化失败${record.failed_initializations}个，来源SHA已保留。`);});
    async function submit(kind) {
      const selected={};for(const role of ["train","validation","test"])if(el("data-"+role).value)selected[role]=el("data-"+role).value;
      const job=await request("coverage/runs",{kind,config:readForm(),dataset_ids:selected,model_id:el("model").value,request_id:"coverage_ui_"+crypto.randomUUID()});runId=job.run_id;await refresh();message("工作台任务已提交："+job.run_id);
    }
    el("start").onclick=()=>guarded(()=>submit("train"));el("evaluate").onclick=()=>guarded(()=>submit("evaluate"));
    el("run").onchange=()=>{runId=el("run").value;runDetail();};
    for(const action of ["stop","resume"])el(action).onclick=()=>guarded(async()=>{if(!runId)throw new Error("请先选择任务");await request("runs/"+encodeURIComponent(runId)+"/"+action,{});await refresh();message(action==="stop"?"停止请求已提交，任务将保存断点。":"断点恢复请求已提交。");});
    const open=()=>{if(location.hash==="#coverage"){for(const page of document.querySelectorAll(".page")){const active=page.id==="page-coverage";page.hidden=!active;page.classList.toggle("active",active);}for(const button of document.querySelectorAll("[data-page]"))button.setAttribute("aria-current",button.dataset.page==="coverage"?"page":"false");}};
    window.addEventListener("hashchange",open);open();setInterval(()=>{if(!el("run").closest(".page").hidden)guarded(runDetail);},7000);
  });
})();
