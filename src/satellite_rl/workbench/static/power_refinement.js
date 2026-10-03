"use strict";
(() => {
  const prefix = "power-refinement-";
  const el = (name) => document.getElementById(prefix + name);
  const pendingKey = "satellite-power-refinement-pending";
  let controls, models = [], scenes = [], runs = [], runId = "", busy = false, reading = false;
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, x => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
  async function request(path, body) {
    const response = await fetch("/api/" + path, body === undefined ? {} : {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "工作台请求失败");
    return data;
  }
  function message(value, bad = false) {
    el("message").textContent = value;
    el("message").className = "notice " + (bad ? "bad" : "muted");
  }
  function input(name, value) { el(name).value = value; }
  function freezePriorityControls() {
    const selected = el("routes").value === "si-priority";
    const fixed = {"si-tolerance":1e-8,"sgm-tolerance":1e-8,"maxiter":300,"maxeval":2000,"max-time":30,
      "si-stall-window":200,"si-stall-tolerance":1e-6,"si-min-evaluations":200,
      "si-max-fraction":0.85,"service-margin-ratio":5e-9};
    for (const [key, value] of Object.entries(fixed)) {
      if (selected) input(key, value);
      el(key).readOnly = selected || key.startsWith("si-stall") ||
        key === "si-min-evaluations" || key === "si-max-fraction" || key === "service-margin-ratio";
    }
  }
  function writeControls() {
    input("routes", controls.methods.join(",") === "sca,sca_si_lp,sca_si_lb" ? "si-priority" :
      controls.methods.length > 3 ? "all" : "legacy");
    for (const key of ["optimizer_seed","population_size","de_mutation","de_crossover","pso_inertia","pso_cognitive","pso_social"]) input(key.replaceAll("_","-"),controls[key]);
    input("action-mode", controls.action_mode);
    input("action-seed", controls.action_seed);
    input("repeats", controls.repeats);
    input("budget-mode", controls.power_budget_mode);
    input("total-power", controls.total_power_w);
    input("protect", String(controls.protect_basic_service));
    input("si-tolerance", controls.si_tolerance);
    input("sgm-tolerance", controls.sgm_tolerance);
    input("maxiter", controls.maxiter);
    input("maxeval", controls.maxeval);
    input("max-time", controls.max_time_seconds);
    for (const key of ["si_stall_window","si_stall_tolerance","si_min_evaluations",
                       "si_max_fraction","service_margin_ratio"]) input(key.replaceAll("_","-"),controls[key]);
    input("discrete-step", controls.discrete_step_w);
    freezePriorityControls();
  }
  function readControls() {
    return {...controls,
      methods:el("routes").value === "si-priority" ? ["sca","sca_si_lp","sca_si_lb"] :
        el("routes").value === "all" ? ["none","discrete","slsqp","sca","de","pso","wmmse"] : ["none","discrete","slsqp"],
      ...Object.fromEntries(["optimizer_seed","population_size","de_mutation","de_crossover","pso_inertia","pso_cognitive","pso_social"].map(key=>[key,Number(el(key.replaceAll("_","-")).value)])),
      action_mode:el("action-mode").value,
      action_seed:Number(el("action-seed").value),
      repeats:Number(el("repeats").value),
      power_budget_mode:el("budget-mode").value,
      total_power_w:Number(el("total-power").value),
      protect_basic_service:el("protect").value === "true",
      si_tolerance:Number(el("si-tolerance").value),
      sgm_tolerance:Number(el("sgm-tolerance").value),
      maxiter:Number(el("maxiter").value),
      maxeval:Number(el("maxeval").value),
      max_time_seconds:Number(el("max-time").value),
      ...Object.fromEntries(["si_stall_window","si_stall_tolerance","si_min_evaluations",
        "si_max_fraction","service_margin_ratio"].map(key=>[key,Number(el(key.replaceAll("_","-")).value)])),
      discrete_step_w:Number(el("discrete-step").value)
    };
  }
  function options(node, records, label, placeholder) {
    const previous = node.value;
    node.replaceChildren(new Option(placeholder, ""));
    for (const record of records) node.add(new Option(label(record), record.id || record.run_id));
    if ([...node.options].some(option => option.value === previous)) node.value = previous;
  }
  function sourceDetail() {
    const model = models.find(row => row.id === el("model").value);
    const scene = scenes.find(row => row.id === el("scene").value);
    el("source").innerHTML = model && scene ?
      `模型：${escape(model.config?.name || model.name)} · ${escape(model.model_files?.[0]?.sha256 || "")}` +
      `<br>考核场景：${escape(scene.name)} · ${escape(scene.count)}景 · ` +
      `${escape((scene.source?.source_dates || scene.source?.snapshot_dates || []).join(", "))}` :
      "请选择G11 T0模型与欧洲test场景；场景、权重和日期在提交时再次校验。";
  }
  function selectedRun() { return runs.find(row => row.run_id === runId); }
  function downloadLink(id, file) {
    const node = el(id);
    node.href = runId ? `/api/runs/${encodeURIComponent(runId)}/download?file=${encodeURIComponent(file)}` : "#";
    node.setAttribute("aria-disabled", runId ? "false" : "true");
  }
  async function detail() {
    if (!runId || reading) return;
    reading = true;
    try {
      const [record, log, result] = await Promise.all([
        request("runs/" + encodeURIComponent(runId)),
        request("runs/" + encodeURIComponent(runId) + "/logs"),
        request("runs/" + encodeURIComponent(runId) + "/results")
      ]);
      const status = record.status || {};
      const done = status.env_step ?? status.completed_method_cases ?? 0;
      const total = status.budget_env_steps ?? record.job?.evaluation_units ?? 0;
      el("status").innerHTML = `状态：${escape(status.stage || "待核验")} · ` +
        `${escape(done)}/${escape(total)} 场景×动作×方法结果` +
        `<br>训练交互：0 · 任务：${escape(runId)}` +
        `<br>源码提交：${escape(record.job?.git_commit || "")}` +
        `<br>当前场景/动作/方法：${escape(status.scene_index ?? "—")}/` +
        `${escape(status.repeat_index ?? "—")}/${escape(status.method ?? "—")}`;
      el("config").textContent = JSON.stringify({job:record.job, config:record.config, status}, null, 2);
      el("logs").textContent = (log.lines || []).join("\n");
      el("results").textContent = JSON.stringify(result.summary || result, null, 2);
      downloadLink("download-report", "power_report.md");
      downloadLink("download-results", "power_results.csv");
      downloadLink("download-beams", "power_beams.csv");
      downloadLink("download-times", "power_scene_times.csv");
      downloadLink("download-cases", "power_cases.jsonl");
      downloadLink("download-iterations", "power_iterations.jsonl");
    } catch (error) { message(error.message, true); }
    finally { reading = false; }
  }
  async function refresh() {
    const [modelData, sceneData, runData] = await Promise.all([
      request("power-refinement/models"), request("power-refinement/scenes"),
      request("power-refinement/runs")
    ]);
    models = modelData.models || []; scenes = sceneData.scenes || []; runs = runData.runs || [];
    options(el("model"), models, row => `${row.config?.name || row.name} · ${row.id}`, "请选择已导入的G11 T0模型");
    options(el("scene"), scenes, row => `${row.name} · ${row.count}景 · test`, "请选择欧洲test场景");
    options(el("run"), runs, row => `${row.name || row.run_id} · ${row.stage}`, "请选择第三阶段任务");
    if (runs.some(row => row.run_id === runId)) el("run").value = runId;
    else if (!runId && runs.length) {runId = runs[0].run_id; el("run").value = runId;}
    sourceDetail();
    await detail();
  }
  async function guarded(callback) {try {await callback();} catch (error) {message(error.message, true);}}
  async function validate() {
    const checked = await request("power-refinement/validate", readControls());
    controls = checked.config;
    writeControls();
    message("配置已校验：所选方法共用同一初始动作；连续功率逐束≥0且总和不超过预算。");
    return controls;
  }
  function stablePending(payload) {
    const encoded = JSON.stringify(payload);
    let pending;
    try {pending = JSON.parse(localStorage.getItem(pendingKey) || "null");} catch {pending = null;}
    if (pending?.encoded === encoded && pending.request_id) return pending;
    pending = {encoded, request_id:"power-refinement-ui-" + crypto.randomUUID()};
    localStorage.setItem(pendingKey, JSON.stringify(pending));
    return pending;
  }
  async function submit() {
    if (busy) return;
    busy = true; el("submit").disabled = true;
    try {
      const config = await validate();
      const model_id = el("model").value, scene_id = el("scene").value;
      if (!model_id || !scene_id) throw new Error("请选择G11 T0模型及欧洲test场景");
      const payload = {config, model_id, scene_id};
      const pending = stablePending(payload);
      const created = await request("power-refinement/runs", {...payload, request_id:pending.request_id});
      if (!created.run_id) throw new Error("服务尚未返回任务编号，请按原请求编号重试");
      runId = created.run_id;
      await refresh();
      localStorage.removeItem(pendingKey);
      message("第三阶段后台任务已登记：" + runId);
    } catch (error) {message(error.message + "；如提交结果不明，再次点击会复用请求编号", true);}
    finally {busy = false; el("submit").disabled = false;}
  }
  async function importModel() {
    const path = el("import-path").value.trim();
    if (!path) throw new Error("请填写T0终点模型目录");
    const created = await request("models/import", {path, name:"G11 T0冻结终点"});
    await refresh();
    if (models.some(row => row.id === created.id)) el("model").value = created.id;
    sourceDetail();
    message("模型已按元数据与权重SHA导入：" + created.id);
  }
  async function action(name) {
    if (!runId) throw new Error("请先选择任务");
    await request("runs/" + encodeURIComponent(runId) + "/" + name, {});
    await refresh();
    message(name === "stop" ? "已向后台提交停止请求。" : "已从工作台提交恢复请求。");
  }
  document.addEventListener("DOMContentLoaded", async () => {
    for (const id of ["refresh","validate","submit","import","stop","resume"]) {
      const callbacks = {refresh,validate,submit,import:importModel,
        stop:()=>action("stop"),resume:()=>action("resume")};
      el(id).addEventListener("click", () => guarded(callbacks[id]));
    }
    el("model").addEventListener("change", sourceDetail);
    el("scene").addEventListener("change", sourceDetail);
    el("run").addEventListener("change", () => {runId = el("run").value; guarded(detail);});
    el("routes").addEventListener("change", freezePriorityControls);
    try {
      controls = (await request("power-refinement/defaults")).defaults;
      writeControls();
      await refresh();
      message("第三阶段配置已加载；正式长评价请在指定远端DQN/CUDA工作台提交。");
    } catch (error) {message(error.message, true);}
    const open = () => {
      if (location.hash !== "#power-refinement") return;
      for (const page of document.querySelectorAll(".page")) {
        const active = page.id === "page-power-refinement";
        page.hidden = !active; page.classList.toggle("active", active);
      }
      for (const button of document.querySelectorAll("[data-page]"))
        button.setAttribute("aria-current", button.dataset.page === "power-refinement" ? "page" : "false");
    };
    window.addEventListener("hashchange", open); open();
    setInterval(() => {if (!el("run").closest(".page").hidden) guarded(detail);}, 7000);
  });
})();
