"""Check local Markdown links and heading anchors in versioned project knowledge."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def headings(path: Path) -> set[str]:
    result = set()
    counts: dict[str, int] = {}
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", path.read_text(), re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        index = counts.get(slug, 0)
        counts[slug] = index + 1
        result.add(f"{slug}-{index}" if index else slug)
    return result


def main() -> int:
    paths = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "--", "*.md"], cwd=ROOT, text=True
    ).splitlines()
    failures = []
    checked = 0
    for name in sorted(set(paths)):
        source = ROOT / name
        if not source.is_file():
            continue
        # Code examples contain historical paths, not navigable documentation links.
        body = re.sub(r"```.*?```", "", source.read_text(), flags=re.DOTALL)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body):
            target = target.strip().strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            destination = (source.parent / unquote(url.path)).resolve() if url.path else source
            checked += 1
            if not destination.exists():
                failures.append(f"{name}: missing {target}")
            elif url.fragment and destination.suffix == ".md":
                if unquote(url.fragment) not in headings(destination):
                    failures.append(f"{name}: missing heading {target}")
    for failure in failures:
        print(failure)
    print(f"Documentation: {checked} local links, {len(failures)} errors")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
