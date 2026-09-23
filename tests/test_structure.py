"""验证结构迁移的依赖边界、导入安全性和已有权重兼容性。"""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "satellite_rl"
AGENT_NAMES = (
    "sac_transformer",
    "sac_cnn",
    "sac_mlp",
    "sac_mlp_matched",
    "sac_cnn_matched",
    "ppo_grouped",
    "ppo_direct",
    "dqn",
)


def runtime_nodes(node):
    """跳过 TYPE_CHECKING 分支，仅检查运行时可能执行的依赖。"""
    yield node
    if isinstance(node, ast.If) and (
        isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        or isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "typing"
        and node.test.attr == "TYPE_CHECKING"
    ):
        children = node.orelse
    else:
        children = ast.iter_child_nodes(node)
    for child in children:
        yield from runtime_nodes(child)


class StructureMigrationTests(unittest.TestCase):
    def run_isolated(self, script):
        """使用非仓库工作目录运行检查，并将潜在实验输出隔离。"""
        with tempfile.TemporaryDirectory(prefix="satellite_rl_test_") as temporary:
            temporary_root = Path(temporary)
            work_directory = temporary_root / "work"
            output_directory = temporary_root / "outputs"
            work_directory.mkdir()
            output_directory.mkdir()
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": str(SOURCE_ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8",
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(temporary_root / "matplotlib"),
                "SATELLITE_RL_OUTPUT_ROOT": str(output_directory),
                "MIGRATION_TEST_REPOSITORY": str(REPOSITORY_ROOT),
            })
            result = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(script)],
                cwd=work_directory,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"隔离检查失败：\n{result.stdout}\n{result.stderr}",
            )
            return (
                list(work_directory.rglob("*")),
                list(output_directory.rglob("*")),
            )

    def test_agents_do_not_import_training_or_environments(self):
        for name in AGENT_NAMES:
            module_path = PACKAGE_ROOT / "agents" / f"{name}.py"
            with self.subTest(agent=name):
                tree = ast.parse(module_path.read_text(encoding="utf-8-sig"))
                for node in runtime_nodes(tree):
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        imports = [node.module or ""]
                        imports += [alias.name for alias in node.names]
                    else:
                        continue
                    for imported_name in imports:
                        parts = imported_name.split(".")
                        self.assertFalse(
                            {"training", "envs"}.intersection(parts)
                            or any(part.startswith("Environment_")
                                   or part.startswith("rl_utils_")
                                   for part in parts),
                            f"{module_path.name} 不应依赖 {imported_name}",
                        )

    def test_all_package_modules_import_without_experiment_outputs(self):
        work_entries, output_entries = self.run_isolated("""
            import importlib
            import pkgutil
            import satellite_rl

            required = {
                "satellite_rl.envs.satellite",
                "satellite_rl.envs.satellite_variant_103",
                "satellite_rl.paths",
            }
            agents = (
                "sac_transformer", "sac_cnn", "sac_mlp",
                "sac_mlp_matched", "sac_cnn_matched",
                "ppo_grouped", "ppo_direct", "dqn",
            )
            for agent in agents:
                required.add("satellite_rl.agents." + agent)
                required.add("satellite_rl.training.train_" + agent)
            discovered = {
                item.name for item in pkgutil.walk_packages(
                    satellite_rl.__path__, satellite_rl.__name__ + "."
                )
            }
            assert required <= discovered, sorted(required - discovered)
            for module_name in sorted(discovered):
                importlib.import_module(module_name)
        """)
        self.assertEqual(work_entries, [], "导入模块不应向工作目录写入实验产物")
        self.assertEqual(output_entries, [], "导入模块不应创建实验日志或输出目录")

    def test_asset_paths_are_independent_of_working_directory(self):
        self.run_isolated("""
            import os
            from pathlib import Path
            from satellite_rl.paths import project_path

            repository = Path(os.environ["MIGRATION_TEST_REPOSITORY"])
            mappings = {
                "cities_china.csv": "data/cities_china.csv",
                "saved_models/sac-t/actor.pth":
                    "models/pretrained/sac_transformer/actor.pth",
                "saved_models/sac-cnn/actor.pth":
                    "models/pretrained/sac_cnn/actor.pth",
                "saved_models/sac-fc/actor.pth":
                    "models/pretrained/sac_mlp/actor.pth",
                "saved_models/ppo/actor.pth":
                    "models/pretrained/ppo_direct/actor.pth",
                "saved_models/dqn/dqn_online.pth":
                    "models/pretrained/dqn/dqn_online.pth",
            }
            for old_path, new_path in mappings.items():
                actual = Path(project_path(old_path)).resolve()
                expected = (repository / new_path).resolve()
                assert actual == expected, (old_path, actual, expected)
                assert actual.is_file(), actual
            output_root = Path(os.environ["SATELLITE_RL_OUTPUT_ROOT"]).resolve()
            log_path = Path(project_path("logs/migration_probe.csv")).resolve()
            assert output_root in log_path.parents, (log_path, output_root)
        """)

    def test_existing_checkpoints_load_into_migrated_networks(self):
        self.run_isolated("""
            import importlib
            import torch
            from satellite_rl.paths import project_path

            torch.set_num_threads(1)
            cases = (
                ("sac_transformer", "PolicyNet", (204, 128, 100, 10, 10),
                 "saved_models/sac-t/actor.pth"),
                ("sac_cnn", "PolicyNet", (204, 128, 100, 10, 10),
                 "saved_models/sac-cnn/actor.pth"),
                ("sac_mlp", "PolicyNet", (204, 128, 100, 10, 10),
                 "saved_models/sac-fc/actor.pth"),
                ("ppo_direct", "PolicyNet", (204,),
                 "saved_models/ppo/actor.pth"),
                ("dqn", "FactorisedDQN", (204,),
                 "saved_models/dqn/dqn_online.pth"),
            )
            for name, class_name, arguments, checkpoint in cases:
                module = importlib.import_module("satellite_rl.agents." + name)
                network = getattr(module, class_name)(*arguments).cpu()
                state_dict = torch.load(
                    project_path(checkpoint), map_location="cpu", weights_only=True
                )
                network.load_state_dict(state_dict, strict=True)
                network.eval()
                with torch.no_grad():
                    heads = network(torch.zeros(1, 204))
                assert len(heads) == 3, name
                for head in heads:
                    assert head.ndim == 2 and head.shape[0] == 1, name
                    assert torch.isfinite(head).all(), name
        """)


if __name__ == "__main__":
    unittest.main()
