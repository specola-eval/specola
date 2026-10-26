# LIBERO demonstration: open the middle drawer of the cabinet

`libero_goal_00_demo.hdf5` is the first human teleoperation demonstration
of LIBERO task `libero_goal:0`, "open the middle drawer of the cabinet",
from the LIBERO datasets by the LIBERO authors (Liu et al., 2023).

- Source: <https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets>,
  revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, file
  `libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5`
  (sha256 `20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737`).
- License: Apache License 2.0, in `LICENSE-Apache-2.0.txt` beside this file.
- Modified: only `data/demo_0/states` (138 flattened MuJoCo states) and the
  file and demonstration attributes are kept. The other 49 demonstrations,
  the camera images, actions, rewards and the other observation arrays are
  dropped, and four provenance attributes are added at the file root. The
  kept states are identical to the source.

Replay it with the specifications in `libero_goal_00_demo_specs.yaml`:

```bash
specola run --sim libero --task libero_goal:0 \
  --queries examples/geometry_queries.yaml \
  --specs examples/libero_goal_00_demo_specs.yaml \
  --mode replay --demo examples/libero_goal_00_demo.hdf5 --web
```
