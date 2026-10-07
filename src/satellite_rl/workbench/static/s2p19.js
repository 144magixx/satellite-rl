"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {catalog: null, scaDefaults: null, runs: [], selectedRun: "", poll: null};
  const copy = (value) => JSON.parse(JSON.stringify(value));
  const safeId = (value) => encodeURIComponent(String(value));
  const sha256 = /^[0-9a-fA-F]{64}$/;

  async function api(path, method = "GET", body) {
    const response = await fetch(path, {
      method,
      headers: method === "POST" ? {"Content-Type": "application/json"} : {},
      body: method === "POST" ? JSON.stringify(body) : undefined,
      cache: "no-store"
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error) throw new Error(data.error || `服务返回 HTTP ${response.status}`);
    return data;
  }

  function requestId() {
    if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
    return `s2p19-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  function message(value, bad = false) {
    const target = $("s2-message");
    target.textContent = value;
    target.className = `notice s2status ${bad ? "bad" : "muted"}`;
  }

  function option(select, value, label) {
    const node = document.createElement("option");
    node.value = String(value);
    node.textContent = String(label);
    select.append(node);
  }

  function sceneOptions(select, scenes, previous, role) {
    select.replaceChildren();
    option(select, "", "请选择已登记的欧洲版场景");
    scenes.filter((row) => (row.profile === "europe_v1" || row.environment_profile?.id === "europe_v1") &&
               (!role || (row.source?.partition_role || row.partition_role) === role))
      .forEach((row) => option(select, row.id, `${row.name || row.id} · ${row.count ?? "?"} 景 · ${row.id}`));
    if (previous && [...select.options].some((row) => row.value === previous)) select.value = previous;
  }

  function configForArm() {
    if (!state.catalog) throw new Error("先读取服务端冻结默认配置");
    const arm = $("s2-arm").value;
    if ($("s2-use-json").checked) {
      const parsed = JSON.parse($("s2-config-json").value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("高级配置必须为JSON对象");
      return parsed;
    }
    const value = copy(state.catalog.defaults?.[arm]);
    if (!value) throw new Error(`服务端未提供${arm}默认配置`);
    value.experiment = {...value.experiment, id: "S2-P19", arm};
    value.environment = {...value.environment, profile: $("s2-profile").value};
    value.training = {...value.training, seed: Number($("s2-seed").value),
                      total_steps: Number($("s2-total").value),
                      pilot_stop_steps: Number($("s2-pilot").value)};
    if (arm === "C1") {
      const path = $("s2-teacher-path").value.trim();
      const manifest = $("s2-teacher-sha").value.trim().toLowerCase();
      const source = $("s2-teacher-source-sha").value.trim().toLowerCase();
      if (!path || !sha256.test(manifest) || !sha256.test(source))
        throw new Error("C1需要教师manifest绝对路径、manifest SHA和完整训练场景集SHA");
      value.training.demonstrations = {mode: "half", manifest_path: path,
                                       manifest_sha256: manifest, source_set_sha256: source};
    }
    return value;
  }

  function i0Config() {
    if ($("s2-i0-use-json").checked) {
      const parsed = JSON.parse($("s2-i0-json").value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("I0高级配置必须为JSON对象");
      return parsed;
    }
    const value = copy(state.catalog?.defaults?.I0);
    if (!value) throw new Error("服务端未提供I0冻结默认配置");
    return value;
  }

  function renderArm() {
    const c1 = $("s2-arm").value === "C1";
    $("s2-teacher-fields").hidden = !c1;
    if (!$('s2-use-json').checked && state.catalog?.defaults?.[$("s2-arm").value]) {
      const preview = copy(state.catalog.defaults[$("s2-arm").value]);
      $("s2-config-json").value = JSON.stringify(preview, null, 2);
    }
  }

  function renderPreparation() {
    const kind = $("s2-preparation-type").value;
    const teacher = kind === "demonstrations";
    for (const id of ["s2-source-path", "s2-source-sha", "s2-prep-seed", "s2-official-assets"])
      $(id).closest("label").hidden = teacher || (id === "s2-official-assets" && kind !== "mother_sources");
    $("s2-prep-scene").closest("label").hidden = !teacher;
    $("s2-selected-sha").closest("label").hidden = !teacher;
    $("s2-prep-note").textContent = teacher
      ? "从完整训练场景集按每母固定规则生成一条贪心轨迹；来源和逐场景SHA由准备任务冻结。"
      : kind === "mother_sources"
        ? "新考核母场景须填写官方资产声明；日期、资格和去重由服务端核验。"
        : "源数据与SHA在提交时由服务端核验；新独立考核不得复用Jan29。";
  }

  function renderI0Mode() {
    if (!state.catalog) return;
    const mode = $("s2-i0-mode").value;
    const role = mode === "train" ? "train" : mode === "gate" ? "development" : "test";
    sceneOptions($("s2-i0-scene"), state.catalog.scenes || [], $("s2-i0-scene").value, role);
    $("s2-i0-validation").closest("label").hidden = mode !== "train";
    $("s2-i0-run").closest("label").hidden = mode !== "evaluate";
    $("s2-i0-repeats").value = mode === "evaluate" ? "3" : "1";
  }

  function parseJsonField(id, label, array = false) {
    const raw = $(id).value.trim();
    if (!raw) throw new Error(`${label}不能为空`);
    const value = JSON.parse(raw);
    if (array ? !Array.isArray(value) : (!value || typeof value !== "object" || Array.isArray(value)))
      throw new Error(`${label}必须为${array ? "JSON数组" : "JSON对象"}`);
    return value;
  }

  function preparationBody() {
    const preparation_type = $("s2-preparation-type").value;
    const body = {request_id: requestId(), preparation: {preparation_type}};
    if (preparation_type === "demonstrations") {
      if (!$("s2-prep-scene").value) throw new Error("先选择完整训练场景集");
      body.scene_id = $("s2-prep-scene").value;
      body.teacher_config = copy(state.catalog.defaults.C0);
      const selected = $("s2-selected-sha").value.trim().toLowerCase();
      if (selected) {
        const hashes = parseJsonField("s2-selected-sha", "教师选取SHA列表", true);
        if (hashes.length !== 576 || hashes.some((value) => typeof value !== "string" || !sha256.test(value)))
          throw new Error("教师选取列表须含576个场景SHA");
        body.selected_sha256 = hashes.map((value) => value.toLowerCase());
      }
    } else {
      const path = $("s2-source-path").value.trim();
      const sourceSha = $("s2-source-sha").value.trim().toLowerCase();
      if (!path || !sha256.test(sourceSha)) throw new Error("源数据路径和SHA-256必须填写");
      body.preparation.source_record = {path, sha256: sourceSha};
      body.preparation.seed = Number($("s2-prep-seed").value);
      body.preparation.environment_profile = $("s2-profile").value;
      if (preparation_type === "mother_sources") {
        body.preparation.official_assets = parseJsonField("s2-official-assets", "官方资产声明", true);
      }
    }
    return body;
  }

  async function submit(button, path, payload, success) {
    button.disabled = true;
    message("工作台正在校验并登记任务…");
    try {
      const data = await api(path, "POST", payload);
      const id = data.run_id || data.job?.run_id || data.id || data.request_id;
      message(`${success}${id ? ` · ${id}` : ""}。请在任务栏核对配置、进度和日志。`);
      await refreshRuns();
      if (id && state.runs.some((run) => run.run_id === id)) await showRun(id);
    } catch (error) {
      message(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function submitPreparation() {
    try { await submit($("s2-prepare"), "/api/s2p19/prepare", preparationBody(), "准备任务已提交"); }
    catch (error) { message(error.message, true); }
  }

  async function validateTraining() {
    const config = configForArm();
    const target = $("s2-config-message");
    target.textContent = "服务端校验中…";
    try {
      const data = await api("/api/validate", "POST", config);
      target.textContent = `服务端校验通过 · 状态维数 ${data.summary?.state_dim ?? "—"} · 算法 ${data.config?.algorithm ?? "—"}`;
      return data.config;
    } catch (error) {
      target.textContent = `校验失败：${error.message}`;
      throw error;
    }
  }

  async function submitTraining() {
    try {
      if (!$("s2-train-scene").value || !$("s2-validation-scene").value)
        throw new Error("先选择欧洲训练与验证场景集");
      const config = await validateTraining();
      const body = {config, scene_id: $("s2-train-scene").value,
                    validation_scene_id: $("s2-validation-scene").value,
                    experiment_id: "S2-P19", request_id: requestId()};
      await submit($("s2-train"), "/api/s2p19/train", body, `${$("s2-arm").value}训练任务已提交`);
    } catch (error) { message(error.message, true); }
  }

  async function submitImprovement() {
    try {
      const mode = $("s2-i0-mode").value;
      if (!$("s2-i0-scene").value) throw new Error("I0需要场景集");
      if (mode === "train" && !$("s2-i0-validation").value) throw new Error("I0训练需要验证场景集");
      if (mode === "evaluate" && !$("s2-i0-run").value) throw new Error("独立评价需要已完成I0模型");
      const body = {mode, i0_config: i0Config(), scene_id: $("s2-i0-scene").value,
                    validation_scene_id: $("s2-i0-validation").value || undefined,
                    model_run_id: $("s2-i0-run").value || undefined,
                    repeats: Number($("s2-i0-repeats").value),
                    action_seed: Number($("s2-i0-action-seed").value), request_id: requestId()};
      await submit($("s2-i0-submit"), "/api/s2p19/improvement", body, `I0 ${mode}任务已提交`);
    } catch (error) { message(error.message, true); }
  }

  async function approvePilot() {
    try {
      const runId = $("s2-gate-run").value;
      const audit_path = $("s2-gate-audit-path").value.trim();
      const audit_sha256 = $("s2-gate-audit-sha").value.trim().toLowerCase();
      if (!runId || !audit_path || !sha256.test(audit_sha256))
        throw new Error("请选择已停试点，并填写审计文件绝对路径及SHA-256");
      await submit($("s2-gate-approve"),
                   `/api/s2p19/runs/${safeId(runId)}/approve-pilot`,
                   {audit_path, audit_sha256}, "科学闸门已由工作台核验并请求同ID续跑");
    } catch (error) { message(error.message, true); }
  }

  function renderSca() {
    const group = $("s2-sca-group").value;
    const partition = $("s2-sca-partition").value;
    const run = state.runs.find((entry) => entry.run_id === $("s2-sca-source").value);
    const inline = run?.kind === "train" && partition === "validation";
    $("s2-sca-step").closest("label").hidden = !inline;
    if (!inline) $("s2-sca-step").value = "";
    if (group === "greedy" || group === "rule" || group === "fixed") {
      $("s2-sca-repeats").value = "1";
      $("s2-sca-action-mode").value = "deterministic";
    } else if (group === "random") {
      $("s2-sca-action-mode").value = "stochastic";
    }
  }

  async function submitSca() {
    try {
      const source_run_id = $("s2-sca-source").value;
      if (!source_run_id) throw new Error("先选择已保存完整初解的来源任务");
      if (!state.scaDefaults) throw new Error("功率细调冻结默认配置尚未读取");
      const partition = $("s2-sca-partition").value;
      const step = $("s2-sca-step").value;
      const run = state.runs.find((entry) => entry.run_id === source_run_id);
      if (run?.kind === "train" && partition === "validation" && !step)
        throw new Error("inline验证初解须选100,000或400,000步节点");
      const config = copy(state.scaDefaults);
      config.methods = ["none", "sca", "sca_si_lp"];
      config.action_mode = $("s2-sca-action-mode").value;
      config.action_seed = Number($("s2-sca-seed").value);
      config.repeats = Number($("s2-sca-repeats").value);
      const body = {source_run_id, source_group: $("s2-sca-group").value,
                    partition, config, request_id: requestId()};
      if (step) body.validation_step = Number(step);
      await submit($("s2-sca-submit"), "/api/power-refinement/runs", body,
                   "已保存初解的统一B任务已提交");
    } catch (error) { message(error.message, true); }
  }

  function addDownload(links, runId, file, label) {
    const anchor = document.createElement("a");
    anchor.href = `/api/runs/${safeId(runId)}/download?file=${safeId(file)}`;
    anchor.textContent = label;
    links.append(anchor);
  }

  function present(value) { return JSON.stringify(value ?? {}, null, 2); }

  async function showRun(runId) {
    state.selectedRun = runId;
    $("s2-run-summary").textContent = `正在读取 ${runId}…`;
    try {
      const [detail, logs, results] = await Promise.allSettled([
        api(`/api/runs/${safeId(runId)}`),
        api(`/api/runs/${safeId(runId)}/logs?limit=120`),
        api(`/api/runs/${safeId(runId)}/results`)
      ]);
      if (detail.status !== "fulfilled") throw detail.reason;
      const d = detail.value, s = d.status || {}, job = d.job || {};
      const progress = Number(s.env_step ?? 0), budget = Number(s.budget_env_steps || 0);
      $("s2-run-summary").textContent = `${job.name || runId} · ${s.stage || "未知阶段"} · ${budget ? `${progress}/${budget} (${(100 * progress / budget).toFixed(1)}%)` : "进度待记录"}${s.stop_reason ? ` · ${s.stop_reason}` : ""}`;
      $("s2-run-summary").className = `notice ${s.stage === "failed" ? "bad" : s.stage === "completed" ? "good" : "muted"}`;
      $("s2-run-config").textContent = present({job, config: d.config, status: s, artifacts: d.artifacts});
      $("s2-run-logs").textContent = logs.status === "fulfilled" ? present(logs.value) : `日志暂不可读：${logs.reason.message}`;
      $("s2-run-results").textContent = results.status === "fulfilled" ? present(results.value) : `结果尚未生成：${results.reason.message}`;
      const links = $("s2-run-links"); links.replaceChildren();
      for (const [file, label] of [["config.json", "配置"], ["status.json", "状态"], ["worker.log", "日志"],
                                  ["results.json", "结果"], ["metrics.jsonl", "训练指标"], ["episodes.jsonl", "逐回合"]])
        addDownload(links, runId, file, label);
    } catch (error) {
      $("s2-run-summary").textContent = error.message;
      $("s2-run-summary").className = "notice bad";
    }
  }

  async function refreshRuns() {
    try {
      const data = await api("/api/s2p19/runs");
      state.runs = Array.isArray(data.runs) ? data.runs : [];
      const holder = $("s2-runs"); holder.replaceChildren();
      if (!state.runs.length) holder.textContent = "当前无S2-P19任务。";
      state.runs.forEach((run) => {
        const card = document.createElement("div"); card.className = "s2card";
        const title = document.createElement("strong"); title.textContent = run.name || run.run_id || "未命名任务";
        const meta = document.createElement("small");
        const done = Number(run.env_step ?? 0), total = Number(run.budget_env_steps || 0);
        meta.textContent = `${run.arm || run.kind || "任务"} · ${run.stage || "待更新"} · ${total ? `${done}/${total}` : "预算待记录"} · ${run.run_id || ""}`;
        const button = document.createElement("button"); button.type = "button"; button.textContent = "查看配置、进度和日志";
        button.addEventListener("click", () => showRun(run.run_id));
        card.append(title, meta, button); holder.append(card);
      });
      const selected = $("s2-i0-run").value;
      $("s2-i0-run").replaceChildren(); option($("s2-i0-run"), "", "选择已登记的I0任务");
      state.runs.filter((run) => run.arm === "I0" || run.route === "I0" ||
                        run.research_stage === "sac_improvement" || String(run.name || "").includes("I0"))
        .forEach((run) =>
        option($("s2-i0-run"), run.run_id, `${run.name || run.run_id} · ${run.stage || "—"}`));
      if (selected && [...$("s2-i0-run").options].some((row) => row.value === selected)) $("s2-i0-run").value = selected;
      const gateSelected = $("s2-gate-run").value;
      $("s2-gate-run").replaceChildren(); option($("s2-gate-run"), "", "选择100,000步已停试点");
      state.runs.filter((run) => run.stage === "stopped" &&
                        (run.arm === "C0" || run.arm === "C1" || /\bC[01]\b/.test(run.name || "")))
        .forEach((run) => option($("s2-gate-run"), run.run_id, `${run.name || run.run_id} · ${run.env_step ?? "?"}步`));
      if (gateSelected && [...$("s2-gate-run").options].some((row) => row.value === gateSelected))
        $("s2-gate-run").value = gateSelected;
      const scaSelected = $("s2-sca-source").value;
      $("s2-sca-source").replaceChildren(); option($("s2-sca-source"), "", "选择已保存完整初解的任务");
      state.runs.filter((run) => (run.kind === "evaluate" ||
                                  (run.kind === "train" && /\bC[01]\b/.test(run.name || ""))) &&
                        (run.stage === "stopped" || run.stage === "completed"))
        .forEach((run) => option($("s2-sca-source"), run.run_id,
          `${run.name || run.run_id} · ${run.stage || "—"} · ${run.run_id}`));
      if (scaSelected && [...$("s2-sca-source").options].some((row) => row.value === scaSelected))
        $("s2-sca-source").value = scaSelected;
      renderSca();
      if (state.selectedRun && state.runs.some((run) => run.run_id === state.selectedRun))
        await showRun(state.selectedRun);
    } catch (error) { message(`任务读取失败：${error.message}`, true); }
  }

  async function refreshCatalog() {
    const data = await api("/api/s2p19/catalog");
    state.catalog = data;
    try {
      const sca = await api("/api/power-refinement/defaults");
      state.scaDefaults = sca.defaults || null;
      if (state.scaDefaults) {
        $("s2-sca-budget").value = `${2 * state.scaDefaults.maxeval}容量调用 / ${2 * state.scaDefaults.max_time_seconds}秒`;
      }
    } catch (error) {
      $("s2-sca-note").textContent = `功率细调默认配置暂不可读：${error.message}`;
    }
    if (!data.defaults?.C0 || !data.defaults?.C1 || !data.defaults?.I0)
      throw new Error("服务端尚未提供C0/C1/I0冻结配置模板");
    if (Array.isArray(data.preparation_types) && data.preparation_types.length) {
      const choices = {dev_validation: "开发/验证场景", mother_sources: "新考核母场景（Feb01/02）",
                       final: "全新独立考核场景", demonstrations: "C1 贪心完整示范"};
      const previous = $("s2-preparation-type").value;
      $("s2-preparation-type").replaceChildren();
      data.preparation_types.forEach((kind) => option($("s2-preparation-type"), kind, choices[kind] || kind));
      if ([...$("s2-preparation-type").options].some((entry) => entry.value === previous))
        $("s2-preparation-type").value = previous;
    }
    const scenes = Array.isArray(data.scenes) ? data.scenes : [];
    for (const [id, role] of [["s2-prep-scene", "train"], ["s2-train-scene", "train"],
                              ["s2-validation-scene", "validation"], ["s2-i0-validation", "validation"]])
      sceneOptions($(id), scenes, $(id).value, role);
    const assets = data.official_assets || data.preparation_defaults?.mother_sources?.official_assets;
    if (assets && !$("s2-official-assets").value.trim())
      $("s2-official-assets").value = JSON.stringify(assets, null, 2);
    if (data.defaults.I0 && !$("s2-i0-json").value.trim())
      $("s2-i0-json").value = JSON.stringify(data.defaults.I0, null, 2);
    renderArm(); renderPreparation(); renderI0Mode(); renderSca();
    $("s2-connection").textContent = `工作台已连接 · 欧洲版 · ${scenes.length}个场景记录`;
  }

  async function refresh() {
    try { await refreshCatalog(); await refreshRuns(); message("已读取服务端场景与任务。提交仍由服务端校验。 "); }
    catch (error) { message(`工作台不可用：${error.message}`, true); $("s2-connection").textContent = "连接失败"; }
  }

  function wire() {
    $("s2-refresh").addEventListener("click", refresh);
    $("s2-runs-refresh").addEventListener("click", refreshRuns);
    $("s2-preparation-type").addEventListener("change", renderPreparation);
    $("s2-arm").addEventListener("change", renderArm);
    $("s2-i0-mode").addEventListener("change", renderI0Mode);
    $("s2-sca-group").addEventListener("change", renderSca);
    $("s2-sca-partition").addEventListener("change", renderSca);
    $("s2-sca-source").addEventListener("change", renderSca);
    $("s2-prepare").addEventListener("click", submitPreparation);
    $("s2-validate").addEventListener("click", () => validateTraining().catch((error) => message(error.message, true)));
    $("s2-train").addEventListener("click", submitTraining);
    $("s2-i0-submit").addEventListener("click", submitImprovement);
    $("s2-gate-approve").addEventListener("click", approvePilot);
    $("s2-sca-submit").addEventListener("click", submitSca);
  }

  wire();
  refresh();
  state.poll = window.setInterval(() => { if (!document.hidden) refreshRuns(); }, 10000);
})();
