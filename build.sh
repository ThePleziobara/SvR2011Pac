#!/bin/sh
# Builds dist/svr_pac-<version>.zip, the file that is installed in Blender.
#   ./build.sh                    (uses the blender on PATH)
#   BLENDER=/path/to/blender ./build.sh
set -e
cd "$(dirname "$0")"
BLENDER="${BLENDER:-blender}"
rm -rf svr_pac/__pycache__
mkdir -p dist
"$BLENDER" --command extension validate svr_pac
"$BLENDER" --command extension build --source-dir svr_pac --output-dir dist
