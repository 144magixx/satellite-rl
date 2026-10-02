"""Third-stage workbench API for frozen G11 M1 power-refinement jobs."""

from __future__ import annotations

from copy import deepcopy
from satellite_rl.power_refinement.config import default_config, validate_config

from .storage import model_input_files, validate_id


G11_M1_RUN_ID = "train_sac_20261001_162205_4ba52a9b"
G11_SOURCE_COMMIT = "85f667cd5731e284c19b2590b5dcf5308a58f234"


def _is_m1(model):
    config = model.get("config") or {}
    network = config.get("network") or {}
    environment = config.get("environment") or {}
    training = config.get("training") or {}
    provenance = model.get("provenance") or {}
    return (not model.get("legacy_type") and config.get("name") == "G11 M1"
            and config.get("algorithm") == "sac" and environment.get("profile") == "europe_v1"
            and environment.get("power_budget_w") == 3000
            and network.get("temporal_mode") == "previous_action"
            and training.get("seed") == 51001 and training.get("total_steps") == 400000
            and training.get("device") == "cuda"
            and provenance.get("run_id") == G11_M1_RUN_ID
            and provenance.get("git_commit") == G11_SOURCE_COMMIT
            and provenance.get("env_step") == 400000)


class PowerRefinementApplication:
    def __init__(self, application):
        self.app = application
        self.store = application.store

    def models(self):
        rows = []
        for model in self.app.models()["models"]:
            if not _is_m1(model):
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

        controls = validate_config(body.get("config", default_config()))
        request_id = body.get("request_id")
        if request_id is not None:
            validate_id(request_id)
            for existing, _ in self.store.jobs():
                if existing.get("request_id") != request_id:
                    continue
                if (existing.get("kind") != "refine" or
                        existing.get("model_id") != body.get("model_id") or
                        existing.get("scene_id") != body.get("scene_id") or
                        existing.get("power_refinement") != controls):
                    raise ValueError("同一请求编号已用于不同的配置")
                return existing
        model = next((row for row in self.models() if row["id"] == body.get("model_id")), None)
        if model is None:
            raise ValueError("请选择已导入工作台且来源可校验的G11 M1冻结模型")
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
        if (provenance.get("run_id") != G11_M1_RUN_ID or
                provenance.get("git_commit") != G11_SOURCE_COMMIT or
                provenance.get("env_step") != 400000):
            raise ValueError("模型权重元数据不是G11 M1固定预算终点")
        trained_sets = provenance.get("scene_sets") or {}
        for role in ("train", "validation"):
            trained = trained_sets.get(role)
            if not trained:
                raise ValueError("M1模型缺少" + role + "来源场景，无法核对日期隔离")
            hashes = {item.get("sha256") for item in trained.get("files", []) if item.get("sha256")}
            if hashes & test_hashes:
                raise ValueError("考核场景与模型训练/验证实际数组重复")
            dates = self.app.scene_dates(trained)
            if not dates:
                raise ValueError("M1模型缺少" + role + "航空来源日期")
            if dates & test_dates:
                raise ValueError("考核航空来源日期与模型训练/验证日期重叠")

        model_config = deepcopy(policy.config)
        config = deepcopy(model_config)
        config["research_stage"] = "power_refinement"
        config["power_refinement"] = controls
        config["name"] = "第三阶段 · G11 M1固定模型功率细调"
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
        return self.store.create_job(config, kind="refine", request_id=body.get("request_id"), extra=extra)

    def api(self, method, parts, query, body):
        if parts == ["defaults"] and method == "GET":
            return {"defaults": default_config(), "methods": [
                {"id": "none", "label": "冻结原始5–50 W档位初始解"},
                {"id": "discrete", "label": "原5 W档位离散细调"},
                {"id": "slsqp", "label": "连续功率优化：SLSQP含需求匹配初始化（p≥0，Σp≤预算）"},
            ]}
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
