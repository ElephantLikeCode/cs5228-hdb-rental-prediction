#!/bin/bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib MPLCONFIGDIR=/private/tmp/5228_mpl
cd "$(dirname "$0")/.."
exec .venv_research/bin/python src/legit_ceiling_20261102.py "$@"
