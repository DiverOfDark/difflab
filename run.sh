#!/bin/sh
pushd frontend
npm run build
popd
uv run main.py
