#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec "$(command -v pixi || echo "$HOME/.pixi/bin/pixi")" run serve
