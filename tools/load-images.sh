#!/bin/bash
# Usage: load-images.sh <archive.tar.zst>   (made by save-images.sh)

set -euo pipefail

ARCHIVE="${1:?Usage: $0 <archive.tar.zst>}"

if [ ! -f "$ARCHIVE" ]; then
  echo "Error: '$ARCHIVE' not found." >&2
  exit 1
fi

if [ -f "$ARCHIVE.sha256" ]; then
  echo "Verifying checksum ..."
  (cd "$(dirname "$ARCHIVE")" && sha256sum -c "$(basename "$ARCHIVE").sha256")
fi

echo "Loading images from $ARCHIVE ..."
zstd -dc "$ARCHIVE" | docker load

docker image ls | grep -E "REPOSITORY|nn-bench-"
