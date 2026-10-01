"""Génère (et exécute, avec --execute) les notebooks autonomes en français + docs/REFERENCES.md.

Usage : python scripts/build_notebooks.py [--execute] [--only 05 06 ...]
Les notebooks sont écrits dans notebooks/ ; aucun n'importe de fichier local (boîte à outils intégrée).
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb import lib  # noqa: E402

ORDER = ["nb01", "nb02", "nb03", "nb04", "nb05", "nb06", "nb07", "nb08", "nb09", "nb10", "nb11", "nb12", "nb13", "nb14", "nb15", "nb16"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="exécute les notebooks et enregistre les sorties")
    p.add_argument("--only", nargs="*", default=None, help="numéros à (re)générer, ex. 05 06")
    a = p.parse_args()
    out = lib.ROOT / "notebooks"; out.mkdir(exist_ok=True)
    lib.write_reference_files()
    for name in ORDER:
        if a.only and name[2:] not in a.only:
            continue
        try:
            mod = importlib.import_module(f"nb.{name}")
        except ModuleNotFoundError:
            continue
        path = out / mod.FILE
        lib.build(path, mod.cells(), mod.TITLE, keep_outputs=not a.execute)
        msg = f"généré : {path.name}"
        if a.execute:
            t0 = time.perf_counter()
            try:
                lib.execute(path); msg += f" (exécuté en {time.perf_counter() - t0:.0f}s)"
            except Exception as exc:  # on continue avec les notebooks suivants
                msg += f" ÉCHEC après {time.perf_counter() - t0:.0f}s : {type(exc).__name__}: {str(exc)[:300]}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
