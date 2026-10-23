"""Print dependencies for the deps-only Docker/Apptainer image."""

import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

with open(sys.argv[1], "rb") as source:
    project = tomllib.load(source)["project"]
requirements = list(project["dependencies"])
for group in ("sim", "vlm", "tracking"):
    requirements.extend(project["optional-dependencies"][group])
print("\n".join(dict.fromkeys(requirements)))
