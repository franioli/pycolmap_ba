# syntax=docker/dockerfile:1
#
# pycolmap_ba on the official COLMAP image (CUDA, ONNX Runtime, GUI).
#
# The official image already ships COLMAP's headers, static libraries, CMake
# config and ONNX Runtime with the CUDA provider, so only the pycolmap bindings
# are compiled here. COLMAP itself is never rebuilt. See docs/docker.md.

# COLMAP 4.2.1. Any release image works (e.g. colmap/colmap:latest): the
# builder reads the version from the image and checks out the matching tag.
ARG COLMAP_IMAGE=colmap/colmap@sha256:c1de51615dff33066cd305071226b6343c1168324ef3bd622f0825e55feb5efe
ARG UV_VERSION=0.13.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

#
# Builder: the pycolmap wheel, linked against the image's COLMAP install.
#
FROM ${COLMAP_IMAGE} AS builder

ARG BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive

# COLMAP's own build dependencies (docker/Dockerfile in its repo), plus the CUDA
# compiler and the CUDA libraries its CMake config asks for. Without them CMake
# silently disables CUDA/GUI and the bindings would be compiled with different
# definitions than the static libraries they link. The CUDA packages follow the
# image's CUDA version. Recommends stay on, as in COLMAP's Dockerfile: the
# exported targets link libiomp5.so, which only arrives as a recommend of
# libmkl-full-dev. The empty opencv4 include dir is the same workaround as
# COLMAP's Dockerfile for Ubuntu's OpenImageIO CMake config.
RUN cuda=$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -) && \
    apt-get update && \
    apt-get install -y \
        build-essential cmake ninja-build git ca-certificates \
        python3 python3-dev \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libopenimageio-dev openimageio-tools libmetis-dev \
        libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
        qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 qt6-svg-dev \
        libcgal-dev libceres-dev libcurl4-openssl-dev libssl-dev libmkl-full-dev \
        cuda-nvcc-${cuda} cuda-cudart-dev-${cuda} libcurand-dev-${cuda} && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/include/opencv4

COPY --from=uv /uv /usr/local/bin/uv

# Check out the source of the exact COLMAP release in the image, and refuse to
# build if its headers differ from the installed ones.
RUN version=$(sed -n 's/^set(PACKAGE_VERSION "\(.*\)")$/\1/p' \
        /usr/local/share/colmap/colmap-config-version.cmake) && \
    echo "COLMAP ${version} in the base image" && \
    git clone --depth 1 --branch "${version}" \
        https://github.com/colmap/colmap.git /src/colmap && \
    cd /usr/local/include/colmap && \
    find . -name '*.h' -not -path './thirdparty/*' | sort > /tmp/headers && \
    test -s /tmp/headers && \
    while read -r header; do \
        cmp -s "${header}" "/src/colmap/src/colmap/${header}" || \
            { echo "header mismatch: ${header}"; exit 1; }; \
    done < /tmp/headers

# CMAKE_CUDA_ARCHITECTURES only has to be valid: the bindings have no CUDA
# sources, and COLMAP's default ("native") fails on a build machine without GPU.
RUN CMAKE_PREFIX_PATH=/usr/local CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    uv build --wheel --python /usr/bin/python3 --out-dir /wheels \
        -C cmake.define.CMAKE_CUDA_ARCHITECTURES=75 \
        /src/colmap

#
# Runtime: the official image plus Python, cuDNN and the project environment.
#
FROM ${COLMAP_IMAGE} AS runtime

# Optional dependency groups from pyproject.toml, comma separated
# (cloud, examples). Empty installs the core only.
ARG EXTRAS=cloud
# cuDNN for ONNX Runtime's CUDA provider. 0 saves ~1 GB on CPU-only machines.
ARG WITH_CUDNN=1
ARG UID=1000
ARG GID=1000
ENV DEBIAN_FRONTEND=noninteractive

# libgomp1 is needed by the open3d wheel.
RUN cuda_major=$(echo "${CUDA_VERSION}" | cut -d. -f1) && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 libmkl-rt libgomp1 \
        $([ "${WITH_CUDNN}" = 1 ] && echo "libcudnn9-cuda-${cuda_major}") && \
    rm -rf /var/lib/apt/lists/*

# Ubuntu images ship a default `ubuntu` user on UID 1000; replace it with one
# matching the host, so files written to bind mounts are owned by the host user.
RUN userdel -r ubuntu 2>/dev/null || true && \
    (getent group "${GID}" || groupadd -g "${GID}" app) && \
    useradd -m -u "${UID}" -g "${GID}" -s /bin/bash app && \
    install -d -o "${UID}" -g "${GID}" /opt/venv /workspace \
        /home/app/.cache/colmap /home/app/.nv

COPY --from=uv /uv /usr/local/bin/uv

# UV_NO_SYNC stops `uv run` from re-resolving pyproject.toml, whose local
# pycolmap wheel path does not exist in the image. COLMAP's CUDA kernels are
# JIT-compiled from PTX on GPUs newer than the release build targets (e.g.
# sm_86); CUDA_CACHE_MAXSIZE keeps the result between runs.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_SYNC=1 \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_CACHE_MAXSIZE=2147483648

USER app
WORKDIR /workspace

# Locked versions of everything except pycolmap, which comes from the builder.
# The path source for pycolmap in pyproject.toml is never resolved here.
COPY --chown=app pyproject.toml uv.lock ./
RUN --mount=type=bind,from=builder,source=/wheels,target=/tmp/wheels \
    extras=$(echo "${EXTRAS}" | tr ',' ' ') && \
    uv export --frozen --no-hashes --no-dev --group test --no-emit-project \
        --no-emit-package pycolmap \
        $(for extra in ${extras}; do printf -- '--extra %s ' "${extra}"; done) \
        -o /tmp/requirements.txt && \
    uv venv /opt/venv --python /usr/bin/python3 && \
    uv pip install -r /tmp/requirements.txt /tmp/wheels/pycolmap-*.whl

# Editable, so a bind mount of the repo on /workspace takes over the sources.
COPY --chown=app src ./src
COPY --chown=app docker/smoke_test.py ./docker/smoke_test.py
RUN uv pip install --no-deps -e .

# Python loads extension modules with RTLD_LOCAL, so MKL cannot resolve its own
# symbols and SIFT brute-force matching aborts. Preloading MKL's dispatcher fixes
# it; set after the installs so the build steps above run without it.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_rt.so

RUN ldd /opt/venv/lib/python3*/site-packages/pycolmap/_core*.so | \
        grep -q libonnxruntime || { echo "pycolmap built without ONNX"; exit 1; } && \
    python -c "import pycolmap, pycolmap_ba; \
assert pycolmap.has_cuda, 'pycolmap built without CUDA'; \
print('pycolmap', pycolmap.__version__, 'has_cuda', pycolmap.has_cuda)"

CMD ["bash"]
