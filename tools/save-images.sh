#!/bin/bash
# Usage: save-images.sh [dest_dir]   (default: ~/docker-cache)
# The archive is several GB

set -euo pipefail

DEST="${1:-$HOME/docker-cache}"
IMAGES=(nn-bench-elixir nn-bench-python)

for img in "${IMAGES[@]}"; do
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    echo "Error: image '$img' not found. Build it first (docker compose build)." >&2
    exit 1
  fi
done

mkdir -p "$DEST"
OUT="$DEST/nn-bench-images-$(date +%F).tar.zst"

echo "Saving ${IMAGES[*]} to $OUT ..."
docker save "${IMAGES[@]}" | zstd -T0 -3 -o "$OUT"

(cd "$DEST" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
ls -lh "$OUT" "$OUT.sha256"
echo "Done. Restore with: tools/load-images.sh $OUT"
