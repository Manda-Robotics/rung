# syntax=docker/dockerfile:1.7
# Rung on upstream NVlabs/RoboLab main 9cce00c (v0.3.2 b6cfcf6 plus the merged Isaac Sim 6
# port, PR #47) with Isaac Sim 6.0.1 and Isaac Lab 3.0, Python 3.12, one NVIDIA GPU.
#
# RoboLab is installed the way its README documents (`uv sync --extra isaac60` against its
# own uv.lock); Isaac Sim and Isaac Lab come from NVIDIA's pip wheels during that sync, so
# by building this image you accept the NVIDIA Isaac Sim license (see README). Rung is not
# pip-installed into RoboLab's environment, because that could replace the numpy Isaac
# pins; it goes on PYTHONPATH instead. The harness needs only numpy and pillow, which the
# RoboLab environment already has.
#
#   docker build -f deploy/robolab/Dockerfile -t rung-robolab .
#   docker run --rm --gpus all -e GEMINI_API_KEY -v "$PWD/out:/out" rung-robolab \
#       python /opt/rung/deploy/robolab/calibrate_tcp.py --out /out/calibration
#
# Host: NVIDIA driver 595 or newer and the NVIDIA Container Toolkit. See
# deploy/robolab/README.md for the probe, calibration and run steps.

ARG ROBOLAB_REVISION=9cce00cadd5dc575fd2b8ea3836f5bdcde574229
ARG ROBOLAB_BASE_REVISION=b6cfcf6ccd2113c4cd1216d35b33024a7d5691da

FROM ubuntu:24.04 AS robolab-src
ARG ROBOLAB_REVISION
ARG ROBOLAB_BASE_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git git-lfs \
    && rm -rf /var/lib/apt/lists/* && git lfs install --system
# The pinned commit must be on upstream main and descend from the v0.3.2 release.
RUN git init -q /src && cd /src && \
    git remote add origin https://github.com/NVlabs/RoboLab.git && \
    git fetch -q origin "${ROBOLAB_REVISION}" "+refs/heads/main:refs/remotes/origin/main" && \
    git -c advice.detachedHead=false checkout -q "${ROBOLAB_REVISION}" && \
    test "$(git rev-parse HEAD)" = "${ROBOLAB_REVISION}" && \
    git merge-base --is-ancestor "${ROBOLAB_BASE_REVISION}" HEAD && \
    git merge-base --is-ancestor HEAD refs/remotes/origin/main && \
    git lfs pull && \
    git rev-parse HEAD > .robolab-revision && \
    rm -rf .git

FROM ghcr.io/astral-sh/uv:0.10.0 AS uv

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia-egl-icd.json \
    OMNI_KIT_ACCEPT_EULA=Y \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    UV_PYTHON=3.12 UV_PYTHON_PREFERENCE=only-system UV_LINK_MODE=hardlink \
    UV_CONCURRENT_DOWNLOADS=2 \
    UV_PROJECT_ENVIRONMENT=/opt/robolab/.venv \
    ROBOLAB_ROOT=/opt/robolab \
    PYTHONPATH=/opt/robolab:/opt/rung/src:/opt/rung/deploy/robolab \
    PATH=/opt/robolab/.venv/bin:$PATH
# Python 3.12, ffmpeg for RoboLab's video writer, and the GL, EGL and Vulkan loaders Isaac
# Sim needs for headless rendering (the driver libraries come from the container toolkit).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3.12 python3.12-venv ffmpeg tini \
    libgl1 libegl1 libglu1-mesa libglvnd0 libvulkan1 vulkan-tools \
    libxt6 libxrandr2 libxcursor1 libxinerama1 libxkbcommon0 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY deploy/robolab/nvidia-egl-icd.json /etc/vulkan/icd.d/nvidia-egl-icd.json
COPY --from=robolab-src /src/ /opt/robolab/
# Cache and environment are on the same Linux filesystem. Hardlinks avoid a second
# full copy of Isaac's extension wheels while installing; removing the cache below
# leaves the environment's links intact.
RUN cd /opt/robolab && uv sync --frozen --python 3.12 --extra isaac60 --no-dev && \
    rm -rf /root/.cache/uv
# Copy runtime inputs explicitly so local recordings, reports and unrelated files
# cannot become image layers when a user builds after running the examples.
COPY src/ /opt/rung/src/
COPY deploy/robolab/ /opt/rung/deploy/robolab/
COPY configs/ /opt/rung/configs/
COPY tasks/ /opt/rung/tasks/
COPY examples/ /opt/rung/examples/
WORKDIR /opt/robolab
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-c", "import robolab, rung; print('robolab', open('/opt/robolab/.robolab-revision').read().strip(), 'rung', rung.__version__)"]
