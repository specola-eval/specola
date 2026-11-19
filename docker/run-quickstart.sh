#!/usr/bin/env bash
# Run only inside the disposable, non-root quickstart container.
set -euo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1
if (( $# == 0 )); then
  cp -R /opt/quickstart-provenance /artifacts/bootstrap
  find /input -type f -print0 | sort -z | xargs -0 -r sha256sum > /artifacts/input-files.sha256
  dpkg-query -W > /artifacts/apt-packages.txt
  "$CHROME" --version > /artifacts/chrome-version.txt
  id > /artifacts/runtime-user.txt
  printf 'lane\tstage\tstatus\texit_code\tseconds\n' > /artifacts/stages.tsv
  failed=0
  for lane in mock libero; do
    stages=(setup assets graphics behavior)
    [[ $lane != mock ]] || stages=(setup readme verification-setup behavior)
    setup_ok=1
    for stage in "${stages[@]}"; do
      if (( ! setup_ok )); then
        printf '%s\t%s\tnot-run\t1\t0\n' "$lane" "$stage" >> /artifacts/stages.tsv
        continue
      fi
      started=$SECONDS
      if bash "$0" "$lane" "$stage" > "/artifacts/$lane-$stage.log" 2>&1; then
        rc=0; state=pass
      else
        rc=$?; state=fail; failed=1
        case $stage in setup|assets|verification-setup) setup_ok=0;; esac
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' "$lane" "$stage" "$state" "$rc" "$((SECONDS-started))" >> /artifacts/stages.tsv
    done
  done
  exit "$failed"
fi
lane=$1; stage=$2
case $lane in mock|libero) ;; *) exit 2;; esac
work=/work/$lane; py=$work/env/bin/python; cli=$work/env/bin/specola
mkdir -p "$work" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd "$work"
wheels=("$work"/wheels/*.whl)
case $stage in
  setup)
    cp -R /input "$work/source"
    if [[ $lane == mock ]]; then
      /usr/bin/python3 -m venv "$work/env"
    else
      /opt/conda/bin/conda create -y -p "$work/env" python=3.12
    fi
    "$py" -m pip install build
    "$py" -m build --wheel --outdir "$work/wheels" "$work/source"
    wheels=("$work"/wheels/*.whl)
    [[ ${#wheels[@]} == 1 && -f ${wheels[0]} ]]
    sha256sum "${wheels[@]}" > "/artifacts/$lane-wheels.sha256"
    extras=web
    [[ $lane == mock ]] || extras=libero,web,report,verify
    "$py" -m pip install "${wheels[0]}[$extras]"
    "$py" -c 'import sys,specola; print(sys.executable); print(sys.version); print(specola.__file__)' > "/artifacts/$lane-installed.txt"
    "$py" -m pip freeze --all > "/artifacts/$lane-packages.txt"
    ;;
  readme)
    mkdir examples
    cp /input/examples/mock_specs.yaml examples/
    mkdir -p /artifacts/readme-mock-results
    ln -s /artifacts/readme-mock-results results
    "$cli" run --sim mock --specs examples/mock_specs.yaml --steps 150 --ui plain
    "$cli" doctor
    ;;
  verification-setup)
    "$py" -m pip install "${wheels[0]}[report,verify]"
    "$py" -m pip freeze --all > "/artifacts/$lane-packages.txt"
    ;;
  assets)
    "$py" -m pip install huggingface-hub==1.25.1
    env -u HF_HUB_OFFLINE "$py" - <<'PY'
import importlib.util, os
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download
root = Path(importlib.util.find_spec("libero").origin).parent / "libero"
assets = Path.home() / ".cache/libero/assets"
config = Path(os.environ["LIBERO_CONFIG_PATH"])
config.mkdir(parents=True, exist_ok=True)
datasets = Path.cwd() / "datasets"
datasets.mkdir()
paths = dict(benchmark_root=str(root), bddl_files=str(root / "bddl_files"),
             init_states=str(root / "init_files"), assets=str(assets), datasets=str(datasets))
configuration = yaml.safe_dump(paths)
(config / "config.yaml").write_text(configuration)
Path('/artifacts/libero-config.yaml').write_text(configuration)
revision = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
print(dict(revision=revision, config=paths), flush=True)
snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset", revision=revision,
                  local_dir=str(assets), max_workers=2)
for category in ("scenes", "articulated_objects", "turbosquid_objects",
                 "stable_scanned_objects", "stable_hope_objects"):
    assert (assets / category).is_dir(), category
from specola.install_verification import native_prerequisites
ok, reason, details = native_prerequisites("libero", dict(os.environ))
print(reason, details, flush=True)
assert ok, reason
PY
    find "$HOME/.cache/libero/assets" -type f -print0 | sort -z | xargs -0 -r sha256sum > /artifacts/libero-assets.sha256
    "$py" -m pip freeze --all > /artifacts/libero-packages.txt
    /opt/conda/bin/conda list -p "$work/env" --explicit > /artifacts/libero-conda-explicit.txt
    ;;
  graphics)
    export HF_HUB_OFFLINE=1
    "$py" - <<'PY'
# Separate probe: these renderer strings do not identify the verifier's context.
import json
from pathlib import Path
import mujoco, numpy as np
from OpenGL import GL
from PIL import Image
ctx = mujoco.GLContext(96, 96)
ctx.make_current()
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size=".2 .2 .2" rgba="1 0 0 1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
scene = mujoco.MjvScene(model, maxgeom=100)
camera = mujoco.MjvCamera()
camera.distance = 2
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
try:
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    viewport = mujoco.MjrRect(0, 0, 96, 96)
    mujoco.mjr_render(viewport, scene, context)
    pixels = np.zeros((96, 96, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(pixels, None, viewport, context)
    Image.fromarray(pixels[::-1]).save('/artifacts/probe.png')
    strings = {name: GL.glGetString(getattr(GL, name)) for name in ('GL_RENDERER', 'GL_VENDOR', 'GL_VERSION')}
    record = dict(scope='Separate probe context only', context=type(ctx).__module__ + '.' + type(ctx).__name__,
                  strings={key: value.decode() if value else None for key, value in strings.items()})
    Path('/artifacts/probe.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    assert all(strings.values()), strings
    assert np.any(pixels != pixels[0, 0]), 'Probe frame is uniform'
finally:
    context.free()
    ctx.free()
PY
    ;;
  behavior)
    export HF_HUB_OFFLINE=1
    options=(--sim "$lane" --timeout 180)
    [[ $lane == mock ]] || options=(--sim libero --demo --timeout 360)
    "$cli" doctor --verify "${options[@]}" --browser "$CHROME" --output-dir "/artifacts/$lane-verification" --json
    ;;
  *) exit 2;;
esac
