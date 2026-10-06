"""前端同步脚本：把 S2-P18 的新选项接进工作台页面（可重复执行）。"""

from pathlib import Path

p = Path("src/satellite_rl/workbench/static/app.js")
text = p.read_text(encoding="utf-8")

anchor = "function experimentEncodingControls(config){"
addition = '''Object.assign(labels,{legacy_repaired_throughput:"旧奖励＋吞吐与频谱使用（S2-P18 V2）",actual:"按实际束数构造（S2-P18 G0）",fixed_100_context:"固定100束上下文",first_token:"读取首个令牌（原读出）",attention_pooling:"注意力池化读出（S2-P18 G0）",film:"FiLM式N条件调制",none_conditioning:"不加N条件",mean_demand:"平均每束需求",total_demand:"全场景总需求"});
Object.assign(fieldNames,{w_throughput:"吞吐项权重 w_throughput",w_spectrum:"频谱使用项权重 w_spectrum",w_shape:"跨束势能整形权重 w_shape",throughput_reference:"吞吐归一化参考量",progress_strata:"回合内进度分层段数",gamma_schedule:"折扣日程（按束数）",context_beam_count:"逐束上下文束数",readout_mode:"编码器读出模式",beam_count_conditioning:"N 条件调制"});
function rewardShapingControls(config){
  const r=config.reward,s=r.shaping||{};
  if(!s||Object.keys(s).length===0)return "";
  const extra=r.preset==="legacy_repaired_throughput"||r.preset==="custom"?`${field("reward.shaping.w_throughput",s.w_throughput??0)}${field("reward.shaping.w_spectrum",s.w_spectrum??0)}`:"";
  return `<h3>S2-P18 奖励分量（默认全部为 0 表示关闭）</h3><div class="field-grid">${extra}${field("reward.shaping.w_shape",s.w_shape??0)}${field("reward.shaping.throughput_reference",s.throughput_reference??"mean_demand",["mean_demand","total_demand"])}</div><p class="field-note">吞吐项为 ΔΣmin(C,D)/参考量（默认“平均每束需求”，与束数无关），频谱项为本步占用槽数÷本组可用槽数，两者只在预设“旧奖励＋吞吐与频谱使用”下生效且权重必须为正；跨束势能整形为 Φ(s)=mean min(C/D,1) 的差分，权重为 0 即关闭，且不改变最优策略。启用后累计回报不能与旧奖励口径直接排名，报告须同时给出原始奖励口径的回报。</p>`;
}
function gammaScheduleControls(config){
  const t=config.training,g=t.gamma_schedule||{mode:"none",default:0.99,by_beam_count:[]};
  const buckets=(g.by_beam_count||[]).map((b)=>`≤${b.max_beam_count}:${b.gamma}`).join("、")||"（无分档）";
  return `<h3>S2-P18 折扣日程与进度分层采样</h3><div class="field-grid">${field("training.gamma_schedule.mode",g.mode||"none",["none","constant","by_beam_count"])}${field("training.gamma_schedule.default",g.default??0.99)}${field("training.progress_strata",t.progress_strata??1)}</div><p class="field-note">mode=none 时使用上方固定折扣（默认 0.99，旧行为逐位不变）；constant 统一取 default；by_beam_count 按实际束数取第一个 max_beam_count ≥ N 的档位。当前分档：${esc(buckets)}（分档明细请在配置 JSON 中编辑，上限须升序）。进度分层段数为 1 时等价于旧采样；设为 4 时每批同时覆盖（数量桶 × 进度段），批大小须能被 桶数×段数 整除，每桶每段的实际采样数与有放回次数写入 status.json 与 metrics.jsonl。</p>`;
}
function entityEncoderControls(config){
  const n=config.network;
  if(n.kind!=="transformer"||n.transformer_layout!=="tsac_natural_context")return "";
  const actual=(n.context_beam_count||"fixed_100")==="actual";
  return `<h3>逐束实体编码（S2-P18 G0）</h3><div class="field-grid">${field("network.context_beam_count",n.context_beam_count||"fixed_100",["fixed_100","actual"])}${actual?field("network.readout_mode",n.readout_mode||"first_token",["first_token","attention_pooling"]):""}${actual?field("network.beam_count_conditioning",n.beam_count_conditioning||"none",["none","film"]):""}</div><p class="field-note">${actual?"逐束上下文按场景实际束数构造（不填充到100），状态维数随 N 变化，训练束数上限必须与场景束数上限一致；读出可选首个令牌或注意力池化，N 条件调制用 FiLM 让同一份权重按规模调整内部表示。实体顺序置换后读出与决策保持一致，占位不参与池化与统计。":"固定100束上下文：与既有对照逐位一致。切换为 actual 需要 layout 或 full 档。"}</p>`;
}
function applyS2P18Preset(kind){
  const c=state.config;if(!c)return;
  c.algorithm="sac";c.network.kind="transformer";
  c.environment.action_mask="joint_feasible";c.environment.action_mode="legacy_clip";
  c.state.demand_visibility="known";
  Object.assign(c.training,{scene_order:"variable_beams",beam_count_mode:"mixed_four_buckets",beam_count_min:50,beam_count_max:200,held_out_beam_counts:[60,90,140,180],seed:51001,batch_size:256,replay_capacity:20000,warmup_steps:2000,update_every:1,gradient_steps:1,alpha_mode:"fixed",fixed_alpha:0.01,gamma:0.99,tau:0.005,actor_lr:1e-5,critic_lr:1e-4,gradient_clip:10,checkpoint_interval:10000,total_steps:100000,progress_strata:1,gamma_schedule:{mode:"none",default:0.99,by_beam_count:[]}});
  c.reward.shaping={w_throughput:0,w_spectrum:0,w_shape:0,throughput_reference:"mean_demand"};
  const s2p18Features=["beam","power","interference","occupancy","beta","progress","remaining_demand","beam_count_normalized","remaining_beams_normalized","power_pressure","group","group_remaining_obligation","remaining_slot_limit"];
  if(kind==="V2"){
    c.network.transformer_layout="tsac_14_variable_context";
    c.network.readout_mode="first_token";c.network.beam_count_conditioning="none";c.network.context_beam_count="fixed_100";
    c.state.features=clone(s2p18Features);
    c.reward.preset="legacy_repaired_throughput";
    c.reward.shaping={w_throughput:50,w_spectrum:1,w_shape:0,throughput_reference:"mean_demand"};
    c.training.progress_strata=4;
  }else if(kind==="V3"){
    c.network.transformer_layout="tsac_14_variable_context";
    c.network.readout_mode="first_token";c.network.beam_count_conditioning="none";c.network.context_beam_count="fixed_100";
    c.state.features=clone(s2p18Features);
    c.reward.preset="legacy_repaired";
    c.reward.shaping={w_throughput:0,w_spectrum:0,w_shape:0.5,throughput_reference:"mean_demand"};
    c.training.gamma_schedule={mode:"by_beam_count",default:0.995,by_beam_count:[{max_beam_count:100,gamma:0.99}]};
    c.training.progress_strata=1;
  }else{
    c.network.transformer_layout="tsac_natural_context";
    c.network.context_beam_count="actual";c.network.readout_mode="attention_pooling";c.network.beam_count_conditioning="film";
    c.state.context_level="layout";
    c.state.features=["beam","power","interference","occupancy","natural_global_context","natural_beam_context","natural_resource_context"];
    c.reward.preset="legacy_repaired";
    c.reward.shaping={w_throughput:0,w_spectrum:0,w_shape:0,throughput_reference:"mean_demand"};
    c.training.progress_strata=1;
  }
  renderConfigForm();saveDraft();scheduleValidation();
  notify("已套用 S2-P18 "+kind+" 预设，请核对束数上限与场景集后提交。");
}
function s2p18PresetButtons(){
  return `<h3>S2-P18 三臂预设（方案18 第二版）</h3><button type="button" data-s2p18="V2">套用 V2 目标与观测修正</button> <button type="button" data-s2p18="V3">套用 V3 信用分配修正</button> <button type="button" data-s2p18="G0">套用 G0 逐束编码原型</button><p class="small muted">V2＝奖励加吞吐/频谱项＋观测修正＋进度分层采样；V3＝同一状态＋按束数折扣＋跨束势能整形；G0＝逐束实体编码（任意 N＋注意力池化＋FiLM）。三臂都复用第二轮冻结场景，任务编号请填 S2-P18。</p>`;
}
function beamCountPanel(status){
  const rows=status?.evaluations_by_beam_count||status?.by_beam_count||status?.beam_count_metrics;
  const steps=status?.beam_count_steps,episodes=status?.beam_count_episodes;
  if(rows&&typeof rows==="object"){
    const body=Object.entries(rows).map(([key,value])=>{const m=value?.metrics||value||{};return `<tr><td>${esc(value?.scene_actions?key+"（"+value.scene_actions+"景）":key)}</td><td>${txt(m.final_si_mean)}</td><td>${txt(m.final_sgm_mean)}</td><td>${txt(m.actual_slots)}</td><td>${txt(m.power_w)}</td></tr>`;}).join("");
    return `<h3>按束数分档的成绩（验证/考核）</h3><table class="table"><thead><tr><th>束数 N</th><th>需求满足率</th><th>供需匹配</th><th>槽数</th><th>功率 W</th></tr></thead><tbody>${body}</tbody></table><p class="small muted">每行按场景束数聚合；含重复动作场景数，不把同场景重复动作当成独立场景。</p>`;
  }
  if(steps){
    const keys=[...new Set([...Object.keys(steps||{}),...Object.keys(episodes||{})])].sort((a,b)=>Number(a)-Number(b));
    const body=keys.map((key)=>`<tr><td>${esc(key)}</td><td>${txt(steps?.[key]??0)}</td><td>${txt(episodes?.[key]??0)}</td></tr>`).join("");
    return `<h3>按束数分档的训练进度</h3><table class="table"><thead><tr><th>束数 N</th><th>交互步</th><th>回合数</th></tr></thead><tbody>${body}</tbody></table><p class="small muted">逐 N 的实际交互步与回合数；进度分层采样的每桶每段配额与有放回次数见 status.json 的 replay_sampled_by_stratum 与 replay_replacement_draws_by_stratum。</p>`;
  }
  return "";
}
'''
assert anchor in text, "anchor missing"
text = text.replace(anchor, addition + anchor, 1)

old_other = '"alpha_mode","fixed_alpha","beam_count_mode","beam_count_min","beam_count_max","held_out_beam_counts"].includes(k)'
new_other = '"alpha_mode","fixed_alpha","beam_count_mode","beam_count_min","beam_count_max","held_out_beam_counts","progress_strata","gamma_schedule"].includes(k)'
assert old_other in text, "other-list anchor missing"
text = text.replace(old_other, new_other, 1)

old_tail = "${temperatureControls(c)}<h3>可变波束数量训练</h3>"
new_tail = "${temperatureControls(c)}${s2p18PresetButtons()}${gammaScheduleControls(c)}${entityEncoderControls(c)}<h3>可变波束数量训练</h3>"
assert old_tail in text, "training tail anchor missing"
text = text.replace(old_tail, new_tail, 1)

old_reward = "${maskOption(c)}`;"
new_reward = "${maskOption(c)}${rewardShapingControls(c)}`;"
assert old_reward in text, "reward tail anchor missing"
text = text.replace(old_reward, new_reward, 1)

old_monitor = "renderCharts(metrics);renderTagBrowser(metrics);"
new_monitor = '$("monitor-meta").insertAdjacentHTML("beforeend",beamCountPanel(s));renderCharts(metrics);renderTagBrowser(metrics);'
assert old_monitor in text, "monitor anchor missing"
text = text.replace(old_monitor, new_monitor, 1)

old_click = "else if(target.dataset.term){"
new_click = "else if(target.dataset.s2p18){applyS2P18Preset(target.dataset.s2p18);}\n" + old_click
assert old_click in text, "click anchor missing"
text = text.replace(old_click, new_click, 1)

p.write_text(text, encoding="utf-8", newline="\n")
print("app.js updated, lines =", text.count("\n") + 1)
