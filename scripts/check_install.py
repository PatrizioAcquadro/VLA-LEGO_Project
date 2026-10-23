"""Build through the sdist and exercise an installed wheel outside the checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile


def run(command, **kwargs):
    subprocess.run([str(arg) for arg in command], check=True, **kwargs)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="vla-install-") as temporary:
        work = Path(temporary)
        dist = work / "dist"
        run(
            [sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir", dist], cwd=root
        )
        shutil.unpack_archive(str(next(dist.glob("*.tar.gz"))), work / "source")
        source = next((work / "source").iterdir())
        run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist],
            cwd=source,
        )
        wheel = next(dist.glob("*.whl"))
        expected = {
            path.relative_to(root / "src").as_posix()
            for path in (root / "src" / "vla_lego").rglob("*")
            if path.is_file()
            and (
                path.suffix in {".py", ".yaml", ".xml", ".obj", ".stl"}
                or path.name == "PROVENANCE.md"
            )
        }
        with ZipFile(wheel) as archive:
            actual = {
                name
                for name in archive.namelist()
                if not name.endswith("/") and ".dist-info/" not in name
            }
        if actual != expected:
            raise RuntimeError(
                f"Wheel contents differ: missing={expected - actual}, extra={actual - expected}"
            )
        print(f"Wheel inventory matches src: {len(expected)} files", flush=True)
        environment = work / "env"
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
        # Reuse the caller's runtime dependencies; project code must come from the wheel.
        site = (
            environment
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        (site / "runtime-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
        python = environment / "bin" / "python"
        run([python, "-m", "pip", "install", "--no-deps", wheel], cwd=work)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        probe = """
import importlib.metadata as metadata
from pathlib import Path
import sys
import vla_lego
from vla_lego import configs, models, sim, tracking
from hydra import compose, initialize_config_module
from vla_lego.sim.asset_loader import SCENES_DIR, load_scene
from vla_lego.sim.bricks.episode import EpisodeManager
from vla_lego.sim.bricks.task import make_goal, evaluate_goal
for module in (vla_lego, configs, models, sim, tracking):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix)), module.__file__
with initialize_config_module(config_module="vla_lego.configs", version_base=None):
    cfg = compose(config_name="config")
assert cfg.model.action_head.action_dim == 17
scenes = sorted(SCENES_DIR.glob("*.xml"))
assert len(scenes) >= 4
for scene in scenes:
    load_scene(scene.name)
episode = EpisodeManager()
goal = make_goal(episode.bricks)
episode.reset(layout=goal)
assert evaluate_goal(episode, goal).success
commands = {entry.name for entry in metadata.distribution("vla-lego").entry_points}
assert commands == {"vla-viewer", "vla-lint-assets", "vla-gen-bricks", "vla-demo", "vla-validate-model"}, commands
print(f"Installed wheel: {len(scenes)} scenes, configuration, tracking, brick physics and five CLIs")
"""
        run([python, "-c", probe], cwd=work, env=env)
        for name in ("vla-viewer", "vla-gen-bricks", "vla-demo", "vla-validate-model"):
            run(
                [environment / "bin" / name, "--help"], cwd=work, env=env, stdout=subprocess.DEVNULL
            )
        run([environment / "bin" / "vla-lint-assets"], cwd=work, env=env)
        run(
            [environment / "bin" / "vla-gen-bricks", "--output-dir", work / "bricks"],
            cwd=work,
            env=env,
        )


if __name__ == "__main__":
    main()
