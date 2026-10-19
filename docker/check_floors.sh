#!/bin/sh
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet --requirement "$root/docker/floors.txt"
cd "$root"
"$work/venv/bin/python" -m pytest -q -p no:cacheprovider
