#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../.." && pwd)
target=$(git -C "${1:?Usage: apply.sh /path/to/cyclo_intelligence}" rev-parse --show-toplevel)
engine=$target/cyclo_brain/policy/lerobot/lerobot_engine/kroc_remote
profiles=$target/shared/shared/robot_configs
profile=$profiles/omx_ai_left_emg_config.yaml
urdf=$profiles/urdf/omx_ai_bimanual.urdf

git -C "$target" apply --check "$here/kroc-remote.patch"
for file in __init__.py client.py; do
  if [[ -e "$engine/$file" ]]; then cmp "$root/kroc_remote/$file" "$engine/$file"; fi
done
for file in cyclo.py emg_file.py window_publisher.py wire.py; do
  if [[ -e "$engine/$file" ]]; then cmp "$here/$file" "$engine/$file"; fi
done
if [[ -e "$profile" ]]; then cmp "$here/omx_ai_left_emg_config.yaml" "$profile"; fi
if [[ -e "$urdf" ]]; then cmp "$here/omx_ai_bimanual.urdf" "$urdf"; fi
git -C "$target" apply "$here/kroc-remote.patch"
install -d "$engine"
install -m 0644 "$root/kroc_remote/"{__init__,client}.py "$engine/"
install -m 0644 "$here/"{cyclo,emg_file,window_publisher,wire}.py "$engine/"
install -m 0644 "$here/omx_ai_left_emg_config.yaml" "$profile"
install -D -m 0644 "$here/omx_ai_bimanual.urdf" "$urdf"
