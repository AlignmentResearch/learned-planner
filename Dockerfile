ARG PYTORCH_CUDA_VERSION=2.3.1-cuda12.1-cudnn8

FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-runtime as main-pre-pip

ARG APPLICATION_NAME=learned-planner
ARG USERID=1001
ARG GROUPID=1001
ARG USERNAME=dev

ENV GIT_URL="https://github.com/AlignmentResearch/${APPLICATION_NAME}"

ENV DEBIAN_FRONTEND=noninteractive
MAINTAINER Adrià Garriga-Alonso <adria@far.ai>
LABEL org.opencontainers.image.source=${GIT_URL}

RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # essential for running. GCC is for Torch triton
    git git-lfs tini build-essential \
    # essential for testing
    libgl-dev libglib2.0-0 zip make \
    # devbox niceties
    curl vim tmux less sudo nano \
    # CircleCI
    ssh \
    # For svg / video rendering
    libcairo2 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository multiverse \
    && apt-get update -q \
    && echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && apt-get install -y ttf-mscorefonts-installer

# Tini: reaps zombie processes and forwards signals
ENTRYPOINT ["/usr/bin/tini", "--"]

# Simulate virtualenv activation
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3 -m venv "${VIRTUAL_ENV}" --system-site-packages \
    && addgroup --gid ${GROUPID} ${USERNAME} \
    && adduser --uid ${USERID} --gid ${GROUPID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "${VIRTUAL_ENV}" "/workspace"

# download Boxoban levels to /training/.sokoban_cache/
RUN mkdir -p "/training/.sokoban_cache/" && chown -R ${USERNAME}:${USERNAME} "/training" \
    && git clone https://github.com/google-deepmind/boxoban-levels "/training/.sokoban_cache/boxoban-levels-master"

USER ${USERNAME}
WORKDIR "/workspace"

FROM main-pre-pip as main

# Install Envpool
ENV ENVPOOL_WHEEL="https://github.com/AlignmentResearch/envpool/releases/download/v0.2.0/envpool-0.8.4-cp310-cp310-linux_x86_64.whl"
RUN pip install "${ENVPOOL_WHEEL}"

# Copy whole repo and install
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install --require-virtualenv -e ".[dev]"

# Run Pyright so its Node.js package gets installed
RUN pyright .

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
