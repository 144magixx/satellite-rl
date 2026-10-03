"""Third-stage workbench API for frozen G11 T0 power-refinement jobs."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from satellite_rl.power_refinement.config import (default_config, validate_config,
                                                  METHOD_LABELS, SI_PRIORITY_METHODS)

from .storage import digest, model_input_files, validate_id


G11_T0_RUN_ID = "train_sac_20261001_162126_33e89e95"
G11_T0_SOURCE_COMMIT = "85f667cd5731e284c19b2590b5dcf5308a58f234"
G11_T0_ENV_STEP = 150000


def _is_t0(model):
    config = model.get("config") or {}
    network = config.get("network") or {}
    environment = config.get("environment") or {}
    training = config.get("training") or {}
    provenance = model.get("provenance") or {}
    return (not model.get("legacy_type") and config.get("name") == "G11 T0"
            and config.get("algorithm") == "sac" and environment.get("profile") == "europe_v1"
            and environment.get("power_budget_w") == 3000
            and network.get("kind") == "transformer"
            and network.get("transformer_layout") == "tsac_natural_context"
            and network.get("temporal_mode") == "none"
            and network.get("graph_mode") == "none"
            and training.get("seed") == 51001 and training.get("total_steps") == G11_T0_ENV_STEP
            and training.get("device") == "cuda"
            and training.get("alpha_mode") == "fixed" and training.get("fixed_alpha") == 0.01
            and provenance.get("run_id") == G11_T0_RUN_ID
            and provenance.get("git_commit") == G11_T0_SOURCE_COMMIT
            and provenance.get("env_step") == G11_T0_ENV_STEP)


class PowerRefinementApplication:
    def __init__(self, application):
        self.app = application
        self.store = application.store

    def models(self):
        rows = []
        for model in self.app.models()["models"]:
            if not _is_t0(model):
                continue
            try:
                files = model_input_files(model["path"])
                if model.get("sha256") and files[0]["sha256"] != model["sha256"]:
                    continue
            except (OSError, ValueError, KeyError):
                continue
            rows.append({**model, "model_files": files})
        return rows

    def scenes(self):
        from .datasets import assert_scene_profile, scene_set_files

        rows = []
        for scene in self.store.records("scenes"):
            if scene.get("research_stage") == "coverage":
                continue
            if (scene.get("source") or {}).get("partition_role") != "test":
                continue
            try:
                assert_scene_profile(scene, "europe_v1")
                if not self.app.scene_dates(scene):
                    continue
                scene_set_files(scene)
            except (OSError, ValueError, KeyError, FileNotFoundError):
                continue
            rows.append(scene)
        return rows

    def submit(self, body):
        from .checkpoints import load_policy
        from .datasets import assert_scene_profile, scene_set_files

        if body.get("source_run_id"):
            return self.submit_saved(body)
        controls = validate_config(body.get("config", default_config()))
        from satellite_rl.power_refinement.config import VARIABLE_BEAM_METHODS
        if controls["methods"] == list(VARIABLE_BEAM_METHODS):
            raise ValueError("可变束数SCA须选择已完成第二阶段分配，不能再次生成初始解")
        request_id = body.get("request_id")
        if request_id is not None:
            validate_id(request_id)
            for existing, _ in self.store.jobs():
                if existing.get("request_id") != request_id:
                    continue
                if (existing.get("kind") != "refine" or
                        existing.get("model_id") != body.get("model_id") or
                        existing.get("scene_id") != body.get("scene_id") or
                        validate_config(existing.get("power_refinement") or {}) != controls):
                    raise ValueError("同一请求编号已用于不同的配置")
                return existing
        priority_deps = None
        priority_dependency_files = None
        if controls["methods"] == list(SI_PRIORITY_METHODS):
            configured = os.environ.get("SATELLITE_RL_PRIORITY_DEPS", "")
            if not configured or not Path(configured).is_absolute() or not Path(configured).is_dir():
                raise ValueError("SI优先任务需要工作台配置绝对且存在的隔离求解依赖目录")
            priority_root = Path(configured).resolve(strict=True)
            priority_deps = str(priority_root)
            priority_dependency_files = []
            for package, version in (("cvxpy", "1.6.5"), ("clarabel", "0.11.1")):
                distribution = priority_root / f"{package}-{version}.dist-info"
                for name in ("METADATA", "RECORD"):
                    file = distribution / name
                    if not file.is_file():
                        raise ValueError(f"SI优先隔离依赖缺少{package} {version}的{name}")
                    priority_dependency_files.append({"path": str(file),
                                                      "sha256": digest(file)})
        model = next((row for row in self.models() if row["id"] == body.get("model_id")), None)
        if model is None:
            raise ValueError("请选择已导入工作台且来源可校验的G11 T0冻结模型")
        scene = self.store.get_record("scenes", body["scene_id"])
        assert_scene_profile(scene, "europe_v1")
        if (scene.get("source") or {}).get("partition_role") != "test":
            raise ValueError("第三阶段正式比较须选择明确test用途的欧洲场景")
        test_dates = self.app.scene_dates(scene)
        if not test_dates:
            raise ValueError("考核场景缺少可核对的航空来源日期")
        test_hashes = {value for _, value in scene_set_files(scene)}

        policy = load_policy(model["path"], config=model["config"], device="cpu")
        provenance = policy.metadata.get("provenance") or {}
        if (provenance.get("run_id") != G11_T0_RUN_ID or
                provenance.get("git_commit") != G11_T0_SOURCE_COMMIT or
                provenance.get("env_step") != G11_T0_ENV_STEP or
                policy.metadata.get("experiment_modes", {}).get("alpha_mode") != "fixed" or
                policy.metadata.get("experiment_modes", {}).get("fixed_alpha") != 0.01):
            raise ValueError("模型权重元数据不是G11 T0固定预算终点")
        trained_sets = provenance.get("scene_sets") or {}
        for role in ("train", "validation"):
            trained = trained_sets.get(role)
            if not trained:
                raise ValueError("T0模型缺少" + role + "来源场景，无法核对日期隔离")
            hashes = {item.get("sha256") for item in trained.get("files", []) if item.get("sha256")}
            if hashes & test_hashes:
                raise ValueError("考核场景与模型训练/验证实际数组重复")
            dates = self.app.scene_dates(trained)
            if not dates:
                raise ValueError("T0模型缺少" + role + "航空来源日期")
            if dates & test_dates:
                raise ValueError("考核航空来源日期与模型训练/验证日期重叠")

        model_config = deepcopy(policy.config)
        config = deepcopy(model_config)
        config["research_stage"] = "power_refinement"
        config["power_refinement"] = controls
        config["name"] = "第三阶段 · G11 T0固定模型功率细调"
        # Workbench status uses this field as a progress denominator. These are
        # evaluation units, never reinforcement-learning interactions.
        config["training"] = deepcopy(config["training"])
        config["training"]["total_steps"] = scene["count"] * controls["repeats"] * len(controls["methods"])
        files = model_input_files(model["path"])
        extra = {
            "name": config["name"], "research_stage": "power_refinement",
            "experiment_role": "formal", "model_id": model["id"],
            "model_path": model["path"], "model_files": files,
            "model_config": model_config, "model_provenance": provenance,
            "scene_id": scene["id"], "scene_path": scene["path"],
            "scene_sets": {"test": scene}, "power_refinement": controls,
            "evaluation_units": config["training"]["total_steps"],
            "evaluation_units_label": "场景×动作采样×方法；非训练交互步",
        }
        if priority_deps is not None:
            extra["priority_solver_deps_path"] = priority_deps
            extra["priority_solver_dependency_files"] = priority_dependency_files
        return self.store.create_job(config, kind="refine", request_id=body.get("request_id"), extra=extra)

    def submit_saved(self, body):
        from .refinement_sources import prepare_source
        from satellite_rl.power_refinement.config import VARIABLE_BEAM_METHODS
        source_job, model_config, dataset, reference = prepare_source(self.app, body)
        controls = validate_config(body.get("config", {}))
        if (controls["methods"] != list(VARIABLE_BEAM_METHODS)
                or controls["power_budget_mode"] != "total"
                or controls["action_mode"] != reference["action_mode"]
                or controls["repeats"] != reference["repeats"]
                or controls["action_seed"] != reference["action_seed"]):
            raise ValueError("S2-P17须保留原动作次数/种子及总3000W的原分配+A+B路线")
        config = deepcopy(model_config)
        config.update(research_stage="power_refinement", power_refinement=controls,
                      name=f"S2-P17 · {source_job['name']} · {reference['group']} · 原分配与SCA")
        config["training"]["total_steps"] = reference["scene_count"] * reference["repeats"] * 3
        return self.store.create_job(config, kind="refine", request_id=body.get("request_id"), extra={
            "research_stage": "power_refinement", "experiment_id": "S2-P17",
            "experiment_role": "formal", "name": config["name"],
            "model_id": source_job["model_id"], "model_path": source_job["model_path"],
            "model_files": source_job["model_files"], "model_config": model_config,
            "model_provenance": source_job.get("model_provenance"),
            "scene_id": source_job["scene_id"], "scene_sets": {"test": dataset},
            "scene_path": dataset["path"], "initial_source": reference,
            "power_refinement": controls, "evaluation_units": config["training"]["total_steps"],
            "evaluation_units_label": "已保存初始分配/中间A/最终B；不新增初始动作",
        })

    def initial_sources(self):
        rows = []
        for job, status in self.store.jobs():
            if (job.get("kind") == "evaluate" and job.get("experiment_id") == "S2-P17"
                    and status.get("stage") == "completed"):
                for group in ["model", *job.get("baselines", [])]:
                    rows.append({"id": job["run_id"] + ":" + group,
                                 "run_id": job["run_id"], "group": group,
                                 "name": job["name"] + " · " + group,
                                 "repeats": job["repeats"] if group == "model" else 1,
                                 "action_mode": job["action_mode"] if group == "model" else "deterministic",
                                 "action_seed": job["action_seed"]})
        return rows

    def api(self, method, parts, query, body):
        if parts == ["initial-sources"] and method == "GET":
            return {"sources": self.initial_sources()}
        if parts == ["defaults"] and method == "GET":
            return {"defaults": default_config(), "methods": [
                {"id": name, "label": label} for name, label in METHOD_LABELS.items()]}
        if parts == ["validate"] and method == "POST":
            return {"config": validate_config(body)}
        if parts == ["models"] and method == "GET":
            return {"models": self.models()}
        if parts == ["scenes"] and method == "GET":
            return {"scenes": self.scenes()}
        if parts == ["runs"]:
            if method == "GET":
                return {"runs": [row for row in self.app.runs()["runs"]
                                 if row.get("research_stage") == "power_refinement"]}
            if method == "POST":
                return self.submit(body)
        raise FileNotFoundError("第三阶段接口不存在")
