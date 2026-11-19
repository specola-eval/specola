#!/usr/bin/env bash
# Build cached bootstrap layers, then perform fresh installations and checks.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
versions=("$@"); (( ${#versions[@]} )) || versions=(22.04 24.04)
for version in "${versions[@]}"; do
  case $version in 22.04|24.04) ;; *) echo "Unsupported Ubuntu: $version" >&2; exit 2;; esac
done
mkdir -p logs/tester-feedback/task-7/runs
out=$(mktemp -d "$repo/logs/tester-feedback/task-7/runs/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf 'Evidence: %s\n' "$out"
printf 'ubuntu\tstage\tstatus\texit_code\tseconds\n' > "$out/rows.tsv"
cidfile=
cleanup() {
  local cid cleanup_rc=0
  if [[ -n $cidfile && -s $cidfile ]]; then
    if cid=$(cat "$cidfile") && [[ $cid =~ ^[0-9a-f]{64}$ ]]; then
      timeout --kill-after=5s 20s docker inspect "$cid" > "${cidfile%/*}/container.json" 2> "${cidfile%/*}/inspect.log" || true
      timeout --kill-after=5s 20s docker stop --time 10 "$cid" >> "${cidfile%/*}/cleanup.log" 2>&1 || true
      timeout --kill-after=5s 20s docker rm --force "$cid" >> "${cidfile%/*}/cleanup.log" 2>&1 || cleanup_rc=1
    else
      echo 'Container identity is unreadable or invalid; cleanup failed.' >> "${cidfile%/*}/cleanup.log"
      cleanup_rc=1
    fi
  fi
  cidfile=
  return "$cleanup_rc"
}
trap 'exit_rc=$?; trap - EXIT; if ! cleanup; then exit_rc=1; fi; exit "$exit_rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for tool in docker timeout; do
  if ! command -v "$tool" > "$out/$tool-path.txt"; then
    echo "Required tool unavailable: $tool" > "$out/prerequisite-failure.txt"
    exit 2
  fi
done
if [[ $(id -u) == 0 || $(uname -m) != x86_64 ]]; then
  echo 'Use a non-root Linux x86_64 account with Docker access.' > "$out/prerequisite-failure.txt"
  exit 2
fi
timeout --kill-after=5s 30s docker version > "$out/docker-version.txt" 2>&1
git rev-parse HEAD > "$out/checkout-head.txt"
status=0
for version in "${versions[@]}"; do
  row=$out/$version
  mkdir -p "$row/artifacts"
  tag=specola-quickstart:$version
  started=$SECONDS
  if timeout --kill-after=30s 2400s docker build --progress=plain \
      --build-arg "UBUNTU_VERSION=$version" --build-arg "RUNTIME_UID=$(id -u)" \
      --iidfile "$row/image-id" -t "$tag" -f docker/Dockerfile . > "$row/build.log" 2>&1; then
    printf '%s\tbuild\tpass\t0\t%s\n' "$version" "$((SECONDS-started))" >> "$out/rows.tsv"
  else
    rc=$?; status=1
    printf '%s\tbuild\tfail\t%s\t%s\n' "$version" "$rc" "$((SECONDS-started))" >> "$out/rows.tsv"
    printf '%s\truntime\tnot-run\t1\t0\n' "$version" >> "$out/rows.tsv"
    continue
  fi
  image=$(cat "$row/image-id")
  if ! timeout --kill-after=5s 30s docker image inspect "$image" > "$row/image.json" 2> "$row/image-inspect.log"; then
    printf '%s\timage-inspect\tfail\t1\t0\n' "$version" >> "$out/rows.tsv"
    status=1
    continue
  fi
  cidfile=$row/container-id
  started=$SECONDS
  if timeout --kill-after=30s 3600s docker run --init --shm-size=1g \
      --stop-timeout=10 --cidfile "$cidfile" \
      --mount "type=bind,source=$row/artifacts,target=/artifacts" "$image" > "$row/run.log" 2>&1; then
    rc=0; state=pass
  else
    rc=$?; state=fail; status=1
  fi
  printf '%s\truntime\t%s\t%s\t%s\n' "$version" "$state" "$rc" "$((SECONDS-started))" >> "$out/rows.tsv"
  if ! cleanup; then
    printf '%s\tcleanup\tfail\t1\t0\n' "$version" >> "$out/rows.tsv"
    status=1
  fi
done
cat "$out/rows.tsv"
exit "$status"
