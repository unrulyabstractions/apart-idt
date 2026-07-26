"""Auto-export helper for package ``__init__.py`` files.

Every package in this repo has a three-line ``__init__.py``::

    from src.common.auto_export import auto_export

    __all__ = auto_export(__file__, __name__, globals())

For each sibling module and subpackage, the names in its ``__all__`` are
re-exported. Modules that fail to import (an optional heavy dependency such as
torch, or a missing credential) are skipped, so installing the analysis stack
never requires the sampling stack.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["auto_export"]


def _import_safe(name: str, package: str) -> Any | None:
    try:
        return importlib.import_module(f".{name}", package=package)
    except ImportError as exc:
        if os.environ.get("AUTO_EXPORT_DEBUG"):
            print(f"[auto_export] skipped {package}.{name}: {exc}", file=sys.stderr)
        return None


def auto_export(init_file: str, package_name: str, globals_dict: dict[str, Any]) -> list[str]:
    """Import sibling modules and subpackages, re-exporting their ``__all__``."""
    directory = Path(init_file).parent
    exported: list[str] = []

    modules = [p.stem for p in sorted(directory.glob("*.py")) if p.name != "__init__.py"]
    packages = [
        p.name
        for p in sorted(directory.iterdir())
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_")
    ]

    for child in modules + packages:
        module = _import_safe(child, package_name)
        if module is None:
            continue
        if child in packages and child not in globals_dict:
            globals_dict[child] = module
            exported.append(child)
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            globals_dict.setdefault(name, obj)
            exported.append(name)

    return exported
