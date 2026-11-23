#!/usr/bin/env python3
"""Static sanity check: compile and import every module of the package plus
the entry-point script. Exits non-zero on the first failure."""

from __future__ import annotations

import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    targets = [os.path.join(HERE, "prepare.py"), os.path.join(HERE, "check.py")]
    for sub in ("plausdl", "tests"):
        for root, dirs, files in os.walk(os.path.join(HERE, sub)):
            dirs[:] = [d for d in dirs if d != "_tmp"]  # scratch, not source
            for f in sorted(files):
                if f.endswith(".py"):
                    targets.append(os.path.join(root, f))
    failed = 0
    for path in targets:
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"COMPILE FAIL {path}: {exc}", file=sys.stderr)
            failed += 1
    if failed:
        return 1
    sys.path.insert(0, HERE)
    import importlib
    for mod in ("plausdl", "plausdl.config", "plausdl.discover", "plausdl.pointcloud",
                "plausdl.normalize", "plausdl.subsample", "plausdl.features",
                "plausdl.pipeline"):
        try:
            importlib.import_module(mod)
        except Exception as exc:
            print(f"IMPORT FAIL {mod}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
    if failed:
        return 1
    print(f"check.py: {len(targets)} files compiled, all modules import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
