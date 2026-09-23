#!/usr/bin/env bash
# Download and convert the NYT-256-angular dataset from ann-benchmarks.
# Requires: python3, numpy, h5py
set -euo pipefail

DATADIR="data/nytimes-256-angular"
HDF5="$DATADIR/nytimes-256-angular.hdf5"

mkdir -p "$DATADIR"

if [ ! -f "$HDF5" ]; then
    echo "Downloading NYT-256-angular..."
    curl -L "https://ann-benchmarks.com/nytimes-256-angular.hdf5" -o "$HDF5"
fi

python3 - "$DATADIR" "$HDF5" << 'PY'
import sys, json
import numpy as np
import h5py

outdir, hdf5 = sys.argv[1], sys.argv[2]
with h5py.File(hdf5) as f:
    np.save(f"{outdir}/base.npy",    f["train"][:].astype("float32"))
    np.save(f"{outdir}/queries.npy", f["test"][:].astype("float32"))
    neighbors = f["neighbors"][:].tolist()
    with open(f"{outdir}/ground_truth.json", "w") as g:
        json.dump([[int(x) for x in row] for row in neighbors], g)
print(f"Saved base ({np.load(outdir+'/base.npy').shape}), "
      f"queries ({np.load(outdir+'/queries.npy').shape}), ground_truth ({len(neighbors)} rows)")
PY

echo "Dataset ready in $DATADIR"
echo ""
echo "SHA256 checksums:"
sha256sum "$DATADIR/base.npy" "$DATADIR/queries.npy"
