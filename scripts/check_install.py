"""Build through the sdist and exercise an installed wheel outside the checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
                or path.name.startswith("LICENSE")
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
        # Embedded Python needs its original binary's library lookup.
        venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(
            environment
        )
        # Reuse the caller's runtime dependencies; project code must come from the wheel.
        site = (
            environment
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        dependencies = [
            path for path in sys.path if Path(path).name in {"site-packages", "dist-packages"}
        ]
        (site / "runtime-dependencies.pth").write_text("\n".join(dependencies) + "\n")
        python = environment / "bin" / "python"
        run([python, "-m", "pip", "install", "--ignore-installed", "--no-deps", wheel], cwd=work)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        probe = (
            f"import sys\nassert sys.version_info[:3] == {sys.version_info[:3]!r}, sys.version\n"
            + """
import importlib.metadata as metadata
from pathlib import Path
import vla_lego
from vla_lego import isaac, pretrained
from vla_lego.isaac.catalog import ASSET_DIRECTORY, catalog
from vla_lego.isaac.task import initial_tower_task
for module in (vla_lego, isaac, pretrained):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix)), module.__file__
blocks = catalog()
assert len(blocks) == 24
assert all(min(block.dimensions_m) > 0 and block.mass_kg > 0 for block in blocks)
assert (ASSET_DIRECTORY / "PROVENANCE.md").is_file()
assert (ASSET_DIRECTORY / "LICENSE_RGB_OBJECTS.txt").is_file()
assert len(initial_tower_task().targets) == 5
commands = {entry.name for entry in metadata.distribution("vla-lego").entry_points}
assert not commands, commands
print("Installed wheel: Isaac/pretrained imports, 24 printable blocks, five-block task and licenses")
"""
        )
        run([python, "-c", probe], cwd=work, env=env)
        for module in ("isaac.smoke", "pretrained.train_pilot", "pretrained.prepare_campaign"):
            run(
                [python, "-m", "vla_lego." + module, "--help"],
                cwd=work,
                env=env,
                stdout=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    main()
