# syntax=docker/dockerfile:1
# Dependencies only. Bind the matching source checkout at /workspace.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-venv libegl1 libgl1 libosmesa6 libglfw3 && rm -rf /var/lib/apt/lists/*
# The base ships the unused spin build tool with a Click constraint incompatible with Hugging Face.
RUN python -m pip uninstall --break-system-packages -y spin
RUN python -m venv --system-site-packages /opt/vla-env
ENV PATH="/opt/vla-env/bin:${PATH}"
COPY pyproject.toml scripts/container-dependencies.py /tmp/
RUN python /tmp/container-dependencies.py /tmp/pyproject.toml > /tmp/requirements.txt && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip check && \
    rm /tmp/pyproject.toml /tmp/container-dependencies.py /tmp/requirements.txt
COPY scripts/container-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "--version"]
