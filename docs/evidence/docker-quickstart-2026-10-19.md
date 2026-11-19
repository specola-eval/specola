# Check the quickstart in Ubuntu containers

Implementation preparation: 2026-10-20. These files are staged for review.
Neither Ubuntu row nor hosted CI has run. Use
`bash docker/check-quickstart.sh 22.04 24.04` after applying the reviewed patch.

The driver builds bootstrap images and performs installations during container
execution. Each row builds separate wheels for Ubuntu system Python and a fresh
Miniforge Python 3.12 environment, outside the input checkout with `PYTHONPATH`
unset. The mock lane runs the README command before adding verifier dependencies.
The installed `doctor --verify` supplies authoring, validation, live output, Stop,
scrubbing, playback, refresh, final-tab closure, port reuse and report checks.
LIBERO uses the packaged goal-0 demo and deterministic verifier specifications,
which differ from the README specification file. Required failures remain failures.

The image selects OSMesa without fallback. A separate MuJoCo scene probe records
its own context class, rendered frame and GL strings; those strings do not
identify the verifier process's renderer. Native frame/scene checks remain required.

Bootstrap inputs are [Miniforge 25.3.1-0](https://github.com/conda-forge/miniforge/releases/tag/25.3.1-0)
with its [published SHA-256](https://conda-forge.org/miniforge/), and
[Chrome 155.0.8059.84, revision 1697595](https://googlechromelabs.github.io/chrome-for-testing/155.0.8059.84.json).
The observed [Storage metadata](https://storage.googleapis.com/storage/v1/b/chrome-for-testing-public/o/155.0.8059.84%2Flinux64%2Fchrome-linux64.zip)
pins generation 1792022877437857, 199014491 bytes and MD5
`ad863e888c8ad9eee36c6bbf21800819`. Setup uses the [Hub 1.25.1 download API](https://raw.githubusercontent.com/huggingface/huggingface_hub/v1.25.1/docs/source/en/guides/download.md)
with [asset snapshot `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`](https://huggingface.co/api/datasets/lerobot/libero-assets/revision/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6).
Metadata does not establish a successful download. Chrome archive SHA-256 remains
unknown until download. Images, apt/transitive packages and solves are recorded,
not frozen.

The image installs the sandbox helper beside Chrome as `chrome-sandbox`, with
root ownership and mode 4755, following [Chromium 155's helper lookup](https://chromium.googlesource.com/chromium/src/+/refs/tags/155.0.8059.84/sandbox/linux/suid/client/setuid_sandbox_host.cc).
Archive membership and sandbox startup remain unverified.

Browser arguments use the existing packaged launcher. [Chrome 155's headless initializer](https://raw.githubusercontent.com/chromium/chromium/155.0.8059.84/chrome/browser/headless/headless_mode_init.cc)
selects `swiftshader-webgl` for those arguments; [its GL feature code](https://chromium.googlesource.com/chromium/src/+/5869fe313571d75152b77813cf0a4ee26a4e6df0/ui/gl/gl_features.cc)
permits that selection when built with SwiftShader. Actual container WebGL remains
unverified.

Artifacts include build/setup/browser logs, results, screenshots, PDF checks,
package lists, wheel and asset hashes, image identity/size and elapsed stages.
Build/run deadlines of 40/60 minutes and 1 GiB shared memory are provisional.
Native Docker layers can be cached; runtime installation and acceptance are fresh.
The independent hosted rows retain artifacts for 14 days, including after failure;
runner loss or a hard job deadline can prevent collection.

Apt resolution, Chrome sandbox/WebGL, asset compatibility, OSMesa rendering,
container timing/size and hosted execution remain unverified. The retained LIBERO
camera scrub failure and native shutdown gates remain Task 8 requirements; this
matrix does not waive them. GPU performance is measured separately from OSMesa.
