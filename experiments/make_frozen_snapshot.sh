#!/usr/bin/env bash
# Recreate the package snapshot behind the reported results (results/raw, run_all.sh,
# run_swap.sh): ../frozen/shufflesnap-bspmatch-20260926. Its sources equal the public release
# tag v0.4.0 except for __version__, so it is rebuilt from that tag in a temporary worktree of
# ../shufflesnap (needs CMake, Ninja and a C++17 compiler; the checkout is left untouched).
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP=${SNAP:-../frozen/shufflesnap-bspmatch-20260926}
if [ -d "$SNAP/src" ]; then
  echo "$SNAP exists"; exit 0
fi
TMP=$(mktemp -d)
trap 'git -C ../shufflesnap worktree remove --force "$TMP/v0.4.0" 2>/dev/null; rm -rf "$TMP"' EXIT
git -C ../shufflesnap worktree add --detach "$TMP/v0.4.0" v0.4.0
(cd "$TMP/v0.4.0" && ./dev_build.sh)
mkdir -p "$SNAP"
cp -r "$TMP/v0.4.0/src" "$SNAP/"
echo "$(git -C ../shufflesnap rev-parse v0.4.0) (tag v0.4.0, rebuilt by make_frozen_snapshot.sh)" > "$SNAP/COMMIT"
echo "$SNAP"
