#!/usr/bin/env bash
# Fetch comparator sources at the exact commits used in the paper (read-only use).
# RasterFairy upstream master is imported directly from third_party/RasterFairy by the
# adapter; BSP-OT is used through the PyPI binding `bspot` (sources kept for reference).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p third_party && cd third_party
fetch() {  # name url commit
  [ -d "$1" ] || git clone -q "$2" "$1"
  git -C "$1" fetch -q origin
  git -C "$1" checkout -q "$3"
  echo "$1 @ $(git -C "$1" rev-parse --short HEAD)"
}
fetch RasterFairy  https://github.com/Quasimondo/RasterFairy.git  610fc842ea196fe5010f3c4786f0d5997528436d
fetch BSP-OT       https://github.com/baptiste-genest/BSP-OT.git  7aa1c013931cc1718a34cebcb121944c13f136e7
fetch bspot-python https://github.com/kenshi84/bspot-python.git   e2636c2b6b1bedb6cded599afe44a1661a1d3c9c
