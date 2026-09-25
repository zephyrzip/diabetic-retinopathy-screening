#!/usr/bin/env bash
set -o errexit

npm ci
python3 -m pip install -r requirements.txt