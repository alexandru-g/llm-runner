# vLLM on a Jetson Orin running JetPack 6 (L4T r36, driver 540 = CUDA 12.6).
#
# PyPI's torch and vLLM are built against CUDA 13, which that driver can't run.
# cuda-compat-orin supplies CUDA 13's user-mode libraries on top of the existing
# kernel driver, so neither the host nor its CUDA 12.6 changes.
#
# Built on the Orin by `llmctl setup vllm --runtime docker`, which takes the
# versions and the package's checksum from engines.json.
FROM ubuntu:22.04

ARG VLLM_VERSION
ARG UV_VERSION
ARG COMPAT_URL
ARG COMPAT_SHA256

# gcc and g++: torch.compile's Triton backend builds kernels with them at startup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl g++ gcc libc6-dev \
        libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/compat.deb "${COMPAT_URL}" \
    && echo "${COMPAT_SHA256}  /tmp/compat.deb" | sha256sum -c - \
    && dpkg-deb -x /tmp/compat.deb / \
    && rm /tmp/compat.deb \
    && ln -s "$(dirname "$(find /usr/local -name libcuda.so.1)")" /usr/local/cuda-compat

# Python lives under /opt: llmctl runs the container as the host's user, not root.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz" \
        | tar -xz --strip-components=1 -C /usr/local/bin \
    && uv venv --python 3.12 /opt/venv \
    && uv pip install -p /opt/venv/bin/python "vllm==${VLLM_VERSION}" \
    && rm -rf /root/.cache

ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-compat
