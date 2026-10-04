"use strict";
(() => {
  let config, catalog, datasets = [], models = [], runs = [], importedStarts = [], runId = "", loading = false;
  const el = (id) => document.getElementById("coverage-" + id);
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (x) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
  async function request(path, body) {
    const response = await fetch("/api/" + path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "请求失败"); return result;
  }
  function message(value, failed = false) { el("message").textContent = value; el("message").className = "notice " + (failed ? "bad" : "muted"); }
  const coverageFields = {"initialization":"initialization", "method":"method", "sweeps":"sweeps", "coordinate-step":"coordinate_step_deg", "max-movement":"max_movement_deg", "demand-floor":"min_coverage_demand", "user-floor":"min_coverage_users", "cluster-iterations":"cluster_iterations"};
  const trainingFields = {"seed":"seed", "total-steps":"total_steps", "num-envs":"num_envs", "rollout-steps":"rollout_steps", "minibatch":"minibatch_size", "epochs":"epochs", "lr":"lr", "validation-interval":"validation_interval", "checkpoint-interval":"checkpoint_interval"};
  const balanceFields = {"route":"route", "action-menu":"action_menu", "step-angle":"step_angle_deg", "goal":"coverage_goal", "subject-cv":"subject_cv_target", "subject-range":"subject_range_tolerance", "subject-count":"min_subject_beams_in_range", "edge-count":"max_edge_beams", "edge-demand-ratio":"edge_demand_ratio", "max-rounds":"max_rounds", "max-decisions":"max_beam_decisions", "repair-decisions":"max_repair_decisions", "repair-evaluations":"max_repair_evaluations", "assignment-passes":"max_assignment_passes", "assignment-attempts":"max_assignment_attempts", "post-move-transfers":"post_move_transfers", "candidate-evaluations":"max_candidate_evaluations", "max-seconds":"max_refinement_seconds"};
  const textBalanceFields = new Set(["route", "action_menu", "coverage_goal"]);
  function balanced() { return config.schema_version === 3; }
  function reconstructed() { return config.schema_version === 4; }
  function writeForm() {
    el("name").value = config.name;
    el("schema").value = String(config.schema_version);
    for (const [id,key] of Object.entries(coverageFields)) el(id).value = config.coverage[key];
    if (balanced()) for (const [id,key] of Object.entries(balanceFields)) el(id).value = config.coverage.balance[key];
    if (reconstructed()) { const r=config.coverage.reconstruction;el("reconstruction-route").value=r.route;el("macro-steps").value=r.max_macro_steps;el("geometry-action").value=r.max_geometry_calls_per_action;el("geometry-total").value=r.max_geometry_calls_total;el("initialization-geometry").value=r.max_initialization_geometry_calls;el("search-width").value=r.search_width;el("split-templates").value=r.split_templates.join(",");el("mechanism-gate").value=config.training.mechanism_gate_step; }
    for (const [id,key] of Object.entries(trainingFields)) el(id).value = config.training[key];
    el("config-json").value = JSON.stringify(config,null,2); applyMode(); renderDatasets();
  }
  function applyMode() {
    const pure=config.coverage.initialization==="empty";
    for(const node of document.querySelectorAll(".coverage-balance-field")) node.hidden = !balanced();
    for(const node of document.querySelectorAll(".coverage-reconstruction-field")) node.hidden = !reconstructed();
    for(const node of document.querySelectorAll(".coverage-start-field")) node.hidden = !(balanced()||reconstructed());
    for(const id of ["method","sweeps","coordinate-step","max-movement","demand-floor","user-floor"])el(id).closest(".field").hidden=reconstructed();
    for(const id of ["coordinate-step","max-movement","sweeps"])el(id).disabled=pure||balanced()||reconstructed();
    el("initialization").disabled=balanced()||reconstructed();el("method").disabled=balanced()||reconstructed();
    for(const option of el("method").options)option.disabled=pure&&option.value!=="ppo";
    el("action-menu").disabled=balanced() && ["prepare","A0","A1","A2","A3","X0","X1","X2","X3","X4"].includes(config.coverage.balance.route);
    el("demand-floor").disabled=balanced()||reconstructed();el("user-floor").disabled=balanced()||reconstructed();
    el("start").disabled=(balanced()&&config.coverage.balance.route!=="ppo")||(reconstructed()&&config.coverage.reconstruction.route!=="P1");
    el("start-heading").textContent=reconstructed()?"S1-03共同覆盖初始化来源":"共同A2起点来源";
    el("start-note").textContent=reconstructed()?"每个分区先提交prepare评价任务；P1训练及R0–R3/P1评价绑定已完成起点。跨实例来源先只读导入，逐SHA冻结。":"先用当前数据分区提交A2准备评价任务。训练、验证及考核分别绑定已完成任务，来源文件和SHA随新任务冻结。";
    el("action-note").textContent=reconstructed()?"S1-03欧洲版固定100束、0.5°全宽。PPO选择完整区域重构宏动作，包括区域/波束与切分模板；STOP立即交出原终态。每动作及每景几何调用有硬预算，覆盖、主体CV与全体束负载独立报告；轨迹最好仅作诊断。":balanced()?"欧洲版固定100束、0.5°全宽；卫星视角格宽与旧经纬度步长不同。四向/八向/多步/业务候选/双束按菜单独立计数；X4双束占两次束处理。覆盖目标、主体CV、例外、早停原因与实际动作在任务中记录。第一阶段未计算的容量、SI、SGM标为不适用。":pure?"纯PPO起始真实激活束为0；从原始正需求航空器位置池中逐次选100个唯一波位。局部格步长、移动范围与移动代价不适用。":"规则/聚类提供100束基础布局，PPO每步保持或局部移动一束；固定0.5°全宽。";
  }
  function readForm() {
    config.name = el("name").value;
    for (const [id,key] of Object.entries(coverageFields)) config.coverage[key] = ["initialization","method"].includes(key) ? el(id).value : Number(el(id).value);
    if (balanced()) {
      for (const [id,key] of Object.entries(balanceFields)) config.coverage.balance[key] = textBalanceFields.has(key) ? el(id).value : Number(el(id).value);
      const b=config.coverage.balance;
      if (["A3","X0","X1","X2","X3","X4"].includes(b.route)) b.action_menu=b.route==="A3"?"X0":b.route;
      config.coverage.initialization="cluster";
      config.coverage.method=b.route==="ppo"?"ppo":b.route==="prepare"?"none":"local_search";
      config.coverage.min_coverage_demand=config.coverage.min_coverage_users=b.coverage_goal==="strict_100"?1:0.999;
      config.coverage.sweeps=b.max_rounds;
    }
    if (reconstructed()) {
      const r=config.coverage.reconstruction;r.route=el("reconstruction-route").value;
      r.max_macro_steps=Number(el("macro-steps").value);r.max_geometry_calls_per_action=Number(el("geometry-action").value);r.max_geometry_calls_total=Number(el("geometry-total").value);r.max_initialization_geometry_calls=Number(el("initialization-geometry").value);r.search_width=Number(el("search-width").value);
      config.coverage.method=r.route==="P1"?"ppo":["prepare","R0"].includes(r.route)?"none":"local_search";
      config.coverage.initialization="cluster";
    }
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
      const training = detail.job.kind === "train";
      const preparing=detail.job.reconstruction_route==="prepare";
      const progress = training ? `${detail.status.env_step||0}/${detail.status.budget_env_steps||0}交互步 · ${detail.status.update_step||0}次更新` : preparing?`共同初始化：${detail.status.prepared_scenes||0}/${detail.status.preparation_total||0}场景`:`评价：${detail.status.evaluated_scenes||0}/${detail.status.evaluation_total||0}场景`;
      const validation = training ? `<p>开发验证：${detail.status.evaluated_scenes||0}/${detail.status.evaluation_total||0}场景</p>` : "";
      const s=results.summary||{};
      const gate=training&&detail.job.reconstruction_route==="P1"?await request("coverage/runs/"+encodeURIComponent(runId)+"/mechanism-gate"):null;
      const balanceInfo=detail.job.balance_route?`<p>路线：${escape(detail.job.balance_route)} · 菜单：${escape(detail.job.balance_action_menu)} · 停止原因：${escape(JSON.stringify(s.stop_reasons||detail.status.stop_reasons||{}))}<br>需求/用户覆盖：${escape(s.coverage_demand)}/${escape(s.coverage_users)} · 主体CV：${escape(s.subject_cv)} · 全体CV：${escape(s.load_cv)} · 例外束：${escape(s.edge_beam_count)}</p>`:"";
      const reconstructionInfo=detail.job.reconstruction_route?`<p>S1-03路线：${escape(detail.job.reconstruction_route)} · 原终态覆盖/主体CV/全体CV：${escape(s.coverage_users)}/${escape(s.subject_cv)}/${escape(s.load_cv)} · 几何调用：${escape(s.geometry_calls||detail.status.geometry_calls||"")}<br>执行实例：${escape(detail.job.execution_target)}${training?` · 机制闸门：${escape(detail.status.stop_reason||"")} ${gate?.proof?.approved?"已记录通过审计":"待审计"}`:""}</p>`:"";
      el("status").innerHTML = `<p>状态：${escape(detail.status.stage)} · ${progress}</p>${validation}${balanceInfo}${reconstructionInfo}<p>源码：${escape(detail.job.git_commit)}<br>GPU：${escape(detail.status.preflight?.gpu||detail.status.gpu||"")}<br>冻结配置与来源保留于任务记录。</p><details><summary>任务配置与状态</summary><pre>${escape(JSON.stringify(detail,null,2))}</pre></details>`;
      el("logs").textContent = (logs.lines||[]).join("\n");
      el("result-download").href="/api/runs/"+encodeURIComponent(runId)+"/results";
      const rawName=detail.job.reconstruction_route==="prepare"?"starts.jsonl":detail.job.reconstruction_route?"results.partial.jsonl":"results.json";
      el("raw-download").href="/api/runs/"+encodeURIComponent(runId)+"/download?file="+encodeURIComponent(rawName);
      el("results").textContent = JSON.stringify(results.summary||results,null,2);
      const list = (metrics.series||metrics.tags||[]); el("curves").innerHTML = list.slice(0,10).map((s) => `<div><p>${escape(s.tag||s.name)}</p>${curve(s)}</div>`).join("") || "暂无原始曲线";
    } catch (error) {message(error.message,true);} finally {loading = false;}
  }
  async function refresh() {
    const [data, modelData, runData, imports] = await Promise.all([request("coverage/datasets"),request("coverage/models"),request("coverage/runs"),request("coverage/reconstruction/imports")]);
    datasets=data.datasets;models=modelData.models;runs=runData.runs;importedStarts=imports.starts||[];renderDatasets();
    for(const role of ["train","validation","test"]){const node=el("start-"+role),previous=node.value;const dataset=datasets.find((d)=>d.id===el("data-"+role).value);let available=[];if(reconstructed()){available=runs.filter((r)=>r.reconstruction_route==="prepare"&&r.dataset_role===role&&r.stage==="completed"&&(!dataset||r.reconstruction_dataset_sha256?.[role]===dataset.sha256)).map((r)=>({id:r.run_id,name:r.name})).concat(importedStarts.filter((r)=>r.partition_role===role&&(!dataset||r.dataset_sha256===dataset.sha256)).map((r)=>({id:r.id,name:r.name+" · 跨实例"})));}else{available=runs.filter((r)=>r.balance_route==="prepare"&&r.dataset_role===role&&r.stage==="completed"&&(!dataset||r.balance_dataset_sha256?.[role]===dataset.sha256)).map((r)=>({id:r.run_id,name:r.name}));}node.innerHTML='<option value="">选择已完成共同起点任务</option>'+available.map((r)=>`<option value="${escape(r.id)}">${escape(r.name)} · ${escape(r.id)}</option>`).join("");if([...node.options].some((o)=>o.value===previous))node.value=previous;}
    const prior=el("model").value;el("model").innerHTML='<option value="">选择第一阶段模型</option>'+models.filter((m)=>m.config?.schema_version===config.schema_version && (!balanced() || m.config?.coverage?.balance?.action_menu===config.coverage.balance.action_menu)).map((m)=>`<option value="${escape(m.id)}">${escape(m.name)} · v${escape(m.config.schema_version)}${balanced()?" · "+escape(m.config.coverage.balance.action_menu):""}</option>`).join("");if([...el("model").options].some((m)=>m.value===prior))el("model").value=prior;
    el("run").innerHTML='<option value="">选择任务</option>'+runs.map((r)=>`<option value="${escape(r.run_id)}">${escape(r.name)} · ${escape(r.stage)}</option>`).join("");if(runs.some((r)=>r.run_id===runId))el("run").value=runId;await runDetail();
  }
  async function guarded(fn) {try {await fn();}catch(error){message(error.message,true);}}
  document.addEventListener("DOMContentLoaded", async () => {
    try {catalog=await request("coverage/catalog");config=catalog.defaults;writeForm();await refresh();message("第一阶段配置已加载；正式任务在当前连接的远端工作台执行。");}catch(error){message(error.message,true);return;}
    el("refresh").onclick=()=>guarded(refresh);
    for(const role of ["train","validation","test"])el("data-"+role).onchange=()=>guarded(refresh);
    el("schema").onchange=()=>{config=JSON.parse(JSON.stringify(el("schema").value==="4"?catalog.reconstruction_defaults:el("schema").value==="3"?catalog.balance_defaults:catalog.defaults));writeForm();refresh().catch((error)=>message(error.message,true));};
    el("route").onchange=()=>{config.coverage.balance.route=el("route").value;if(["A3","X0","X1","X2","X3","X4"].includes(el("route").value))config.coverage.balance.action_menu=el("route").value==="A3"?"X0":el("route").value;config.coverage.method=el("route").value==="ppo"?"ppo":el("route").value==="prepare"?"none":"local_search";writeForm();};
    el("reconstruction-route").onchange=()=>{const route=el("reconstruction-route").value;config.coverage.reconstruction.route=route;config.coverage.method=route==="P1"?"ppo":["prepare","R0"].includes(route)?"none":"local_search";writeForm();};
    el("goal").onchange=()=>{config.coverage.balance.coverage_goal=el("goal").value;config.coverage.min_coverage_demand=config.coverage.min_coverage_users=el("goal").value==="strict_100"?1:0.999;writeForm();};
    el("initialization").onchange=()=>{config.coverage.initialization=el("initialization").value;if(config.coverage.initialization==="empty"){config.coverage.method="ppo";config.coverage.sweeps=1;}else{config.coverage.sweeps=2;}writeForm();};
    el("validate").onclick=()=>guarded(async()=>{const result=await request("coverage/validate",readForm());config=result.config;writeForm();message(reconstructed()?`S1-03校验通过：${result.summary.route}，最多${result.summary.max_macro_steps}次完整宏动作，单次/每景几何调用上限${result.summary.max_geometry_calls_per_action}/${result.summary.max_geometry_calls_total}。`:`校验通过：${typeof result.summary.state_dim==="number"?result.summary.state_dim+"维状态":"状态维数以实际环境记录为准"}，${result.summary.action_dim}个候选动作。`);});
    el("apply-json").onclick=()=>guarded(async()=>{const result=await request("coverage/validate",JSON.parse(el("config-json").value));config=result.config;writeForm();message("完整配置已校验并应用。");});
    el("save").onclick=()=>guarded(async()=>{const saved=await request("coverage/configs",{config:readForm()});message("方案已保存："+saved.id);});
    el("import-data").onclick=()=>guarded(async()=>{message("正在按基础布局准备逐机数据…");const record=await request("coverage/datasets/import",{path:el("data-path").value,partition_role:el("data-role").value,coverage:readForm().coverage});await refresh();el("data-"+record.partition_role).value=record.id;message(`已保存${record.count}个场景；初始化失败${record.failed_initializations}个，来源SHA已保留。`);});
    el("import-dataset").onclick=()=>guarded(async()=>{const row=await request("coverage/reconstruction/import-dataset",{source_target:el("import-data-target").value,source_dataset_id:el("import-dataset-id").value,partition_role:el("import-data-role").value});await refresh();el("data-"+row.partition_role).value=row.id;message(`数据已只读导入：${row.id} · SHA ${row.sha256}`);});
    el("import-start").onclick=()=>guarded(async()=>{const row=await request("coverage/reconstruction/import-start",{source_target:el("import-target").value,source_run_id:el("import-run-id").value,partition_role:el("import-role").value});await refresh();message(`共同起点已只读导入：${row.id} · SHA ${row.sha256}`);});
    el("import-model").onclick=()=>guarded(async()=>{const row=await request("coverage/reconstruction/import-model",{source_target:el("import-target").value,source_run_id:el("import-run-id").value});await refresh();message(`P1模型已只读导入：${row.id} · SHA ${row.sha256}`);});
    async function submit(kind) {
      const selected={};for(const role of ["train","validation","test"])if(el("data-"+role).value)selected[role]=el("data-"+role).value;
      const starts={};for(const role of ["train","validation","test"])if(el("start-"+role).value)starts[role]=el("start-"+role).value;
      const job=await request("coverage/runs",{kind,config:readForm(),dataset_ids:selected,dataset_role:el("dataset-role").value,start_run_ids:starts,model_id:el("model").value,experiment_role:el("purpose").value,request_id:"coverage_ui_"+crypto.randomUUID()});runId=job.run_id;await refresh();message("工作台任务已提交："+job.run_id);
    }
    el("start").onclick=()=>guarded(()=>submit("train"));el("evaluate").onclick=()=>guarded(()=>submit("evaluate"));
    el("run").onchange=()=>{runId=el("run").value;runDetail();};
    for(const action of ["stop","resume"])el(action).onclick=()=>guarded(async()=>{if(!runId)throw new Error("请先选择任务");await request("runs/"+encodeURIComponent(runId)+"/"+action,{});await refresh();message(action==="stop"?"停止请求已提交，任务将保存断点。":"断点恢复请求已提交。");});
    const open=()=>{if(location.hash==="#coverage"){for(const page of document.querySelectorAll(".page")){const active=page.id==="page-coverage";page.hidden=!active;page.classList.toggle("active",active);}for(const button of document.querySelectorAll("[data-page]"))button.setAttribute("aria-current",button.dataset.page==="coverage"?"page":"false");}};
    window.addEventListener("hashchange",open);open();setInterval(()=>{if(!el("run").closest(".page").hidden)guarded(runDetail);},7000);
  });
})();
