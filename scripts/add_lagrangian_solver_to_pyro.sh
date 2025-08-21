#!/usr/bin/env bash
set -euo pipefail
PYRO_ROOT="${1:-.}"
echo "[INFO] Copying compressible_lagrangian into $PYRO_ROOT/pyro"
if command -v rsync >/dev/null 2>&1; then
  rsync -av --delete "./pyro/compressible_lagrangian" "$PYRO_ROOT/pyro/"
else
  python - "$PYRO_ROOT" <<'PY'
import os, sys, shutil, pathlib
root = pathlib.Path(sys.argv[1])
dst = root / "pyro" / "compressible_lagrangian"
src = pathlib.Path("pyro/compressible_lagrangian")
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst)
print("[OK] Copied via Python copytree")
PY
fi
echo "[INFO] Attempting to register solver in pyro/pyro.py"
PFILE="$PYRO_ROOT/pyro/pyro.py"
if [ -f "$PFILE" ]; then
python - "$PFILE" <<'PY'
import sys
pfile=sys.argv[1]
s=open(pfile,'r',encoding='utf-8').read()
if 'compressible_lagrangian' in s:
    print("[SKIP] Entry already present"); raise SystemExit
if 'import importlib' not in s:
    s = 'import importlib\n' + s
s += "\n" + """
elif solver == "compressible_lagrangian":
    mod = importlib.import_module("pyro.compressible_lagrangian.simulation")
    self.simulation = mod.Simulation(self.rp)
""" + "\n"
open(pfile,'w',encoding='utf-8').write(s)
print("[OK] Registered compressible_lagrangian")
PY
else
  echo "[WARN] Could not find $PFILE; please add factory mapping manually."
fi
echo "[DONE] Lagrangian solver added. Commit your changes."
