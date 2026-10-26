# Fresh-environment install of specola[robodojo], 2026-10-07

Second end-to-end install, at commit `72c8fe9` of branch `adapter-parity`.
Logs: `results/parity-audit-2026-10-02/logs/fresh-robodojo/` (not committed).

## What "fresh" means here

RoboDojo's own `scripts/install.sh` builds a conda environment named
`RoboDojo` (Python 3.11, Isaac Sim 5.1 from NVIDIA's index, Isaac Lab
from its `third_party/`, torch cu128, curobo; 22 GB). The user path for
Specola is to install RoboDojo per its documentation, then
`pip install specola[robodojo]` inside that environment. So the check
cloned the working `RoboDojo` environment, removed the editable Specola
from the clone, and installed the repository non-editably with the new
`robodojo` extra. The RoboDojo stack itself was not reinstalled from
zero.

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 | `conda create --clone RoboDojo -n fresh-robodojo` | ok, with a clone artifact: duplicate `pip`, `setuptools`, `wheel` and `packaging` dist-infos (conda's versions beside the pip-installed pins), which broke `pip`. Removed and re-pinned to RoboDojo's versions (`pip` via `ensurepip`, `packaging==23.0`, `wheel==0.45.1`, `setuptools==81.0.0`). Not a Specola matter |
| 2 | `pip uninstall specola` | ok |
| 3 | `pip install <repo>[robodojo]` (non-editable) | ok |
| 4 | `pip check` | the same three conflicts as the original environment (`isaacsim-kernel` coverage, `rtamt` antlr, `isaaclab` starlette); Specola added none |
| 5 | `specola run --sim mock ... --steps 150 --ui plain` | `150 steps · 7.500 s simulator time · completion complete` |
| 6 | `specola discover --sim robodojo --task stack_bowls --queries examples/robodojo_queries.yaml` | ok |
| 7 | `specola run --sim robodojo --task stack_bowls --queries ... --specs examples/robodojo_specs.yaml --mode hold --steps 20 --ui plain --web 0` | `20 steps · 0.080 s simulator time · completion complete`; dashboard served, `GET /` 200; recording with frames, geometry, meshes and poses written; process exited 3 s after SIGINT to its group |

Every RoboDojo command ran with `OMNI_KIT_ACCEPT_EULA=YES` and
`ROBODOJO_ROOT` pointing at the checkout, and the dashboard address
was read with `PYTHONUNBUFFERED=1`. None of these are in the README yet.

## Findings

- The `robodojo` extra (dashboard and report dependencies) installs
  into the RoboDojo stack without moving any of its pins. Before this
  extra, `report` was refused in that environment (#134); a fresh
  install with the extra has pandas and pyarrow.
- `conda create --clone` of an environment that pip has mutated is not
  a faithful copy. For install checks, prefer the producer's own
  installer into a new environment name, or a container.
