ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}
ARG UBUNTU_VERSION
ARG RUNTIME_UID=1000
ARG DEBIAN_FRONTEND=noninteractive
# Dependency documentation and observed archive metadata: Task 7's dated evidence.
RUN case "$UBUNTU_VERSION" in \
      22.04) desktop='libasound2 libatk-bridge2.0-0 libglib2.0-0 libgtk-3-0';; \
      24.04) desktop='libasound2t64 libatk-bridge2.0-0t64 libglib2.0-0t64 libgtk-3-0t64';; \
      *) exit 2;; esac \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl unzip build-essential python3-venv python3-dev \
      libosmesa6 libegl1 libegl-mesa0 libgl1 libgles2 libglvnd0 \
      fonts-liberation libnss3 libnspr4 libgbm1 libdbus-1-3 libexpat1 \
      libcairo2 libpango-1.0-0 libfontconfig1 libx11-6 libxcb1 libxcomposite1 \
      libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 \
      libxss1 libxtst6 poppler-utils $desktop \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/quickstart-provenance \
    && curl -fL --retry 3 https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh -o /tmp/miniforge.sh \
    && echo '376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768  /tmp/miniforge.sh' | sha256sum -c - \
    && sha256sum /tmp/miniforge.sh > /opt/quickstart-provenance/miniforge.sha256 \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
# Select the observed object generation and validate its bytes before unpacking.
RUN curl -fL --retry 3 -D /opt/quickstart-provenance/chrome-download.headers \
      'https://storage.googleapis.com/download/storage/v1/b/chrome-for-testing-public/o/155.0.8059.84%2Flinux64%2Fchrome-linux64.zip?generation=1792022877437857&alt=media' -o /tmp/chrome.zip \
    && test "$(stat -c %s /tmp/chrome.zip)" = 199014491 \
    && echo 'ad863e888c8ad9eee36c6bbf21800819  /tmp/chrome.zip' | md5sum -c - \
    && sha256sum /tmp/chrome.zip > /opt/quickstart-provenance/chrome.sha256 \
    && unzip -q /tmp/chrome.zip -d /opt/chrome && rm /tmp/chrome.zip \
    && install -o root -g root -m 4755 /opt/chrome/chrome-linux64/chrome_sandbox /opt/chrome/chrome-linux64/chrome-sandbox \
    && ldd /opt/chrome/chrome-linux64/chrome > /opt/quickstart-provenance/chrome.ldd \
    && ! grep -q 'not found' /opt/quickstart-provenance/chrome.ldd
RUN test "$RUNTIME_UID" -gt 0 \
    && useradd --non-unique --uid "$RUNTIME_UID" --create-home quickstart \
    && mkdir /work /artifacts && chown quickstart /work /artifacts
ENV PYTHONNOUSERSITE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    LIBGL_ALWAYS_SOFTWARE=true CHROME=/opt/chrome/chrome-linux64/chrome \
    NUMBA_CACHE_DIR=/home/quickstart/.cache/specola/numba \
    MPLCONFIGDIR=/home/quickstart/.cache/specola/matplotlib \
    LIBERO_CONFIG_PATH=/home/quickstart/.libero \
    CONDA_PKGS_DIRS=/home/quickstart/.cache/conda/pkgs
COPY . /input
USER quickstart
WORKDIR /work
ENTRYPOINT ["bash", "/input/docker/run-quickstart.sh"]
