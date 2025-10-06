#!/usr/bin/env python3
"""
Check if this Mac supports Apple's Metal.

Strategy:
1) Try the official Metal API via PyObjC (preferred).
2) Fallback to `system_profiler` (works out-of-the-box).
Exit code 0 = Metal available, 1 = Metal not available / not macOS.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Dict, Any

RESULT: Dict[str, Any] = {
    "platform": platform.platform(),
    "is_macos": platform.system() == "Darwin",
    "method": None,
    "metal_supported": False,
    "details": {},
    "notes": [],
}

def print_and_exit(ok: bool):
    RESULT["metal_supported"] = bool(ok)
    print(json.dumps(RESULT, indent=2, sort_keys=True))
    sys.exit(0 if ok else 1)

def try_pyobjc() -> bool:
    try:
        # PyObjC Metal bridge (pip install pyobjc; comes preinstalled on many Macs)
        from Metal import MTLCreateSystemDefaultDevice, MTLCopyAllDevices  # type: ignore
    except Exception as e:
        RESULT["notes"].append(f"PyObjC/Metal import failed: {e!r}")
        return False

    RESULT["method"] = "pyobjc"
    try:
        devices = MTLCopyAllDevices()  # returns a list of MTLDevice
        names = []
        for d in devices:
            try:
                # .name() is an Objective-C method; PyObjC exposes it as a callable
                names.append(str(d.name()))
            except Exception:
                names.append("<unknown device name>")
        RESULT["details"]["metal_devices"] = names
        return len(devices) > 0
    except Exception as e:
        RESULT["notes"].append(f"PyObjC Metal probe error: {e!r}")
        return False

def _run_profiler(datatype: str) -> str:
    try:
        return subprocess.check_output(
            ["system_profiler", "-detailLevel", "mini", datatype],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""

def try_system_profiler() -> bool:
    RESULT["method"] = "system_profiler"
    if shutil.which("system_profiler") is None:
        RESULT["notes"].append("system_profiler not found in PATH.")
        return False

    # Try the dedicated Metal data type (exists on many recent macOS versions),
    # then fall back to display info which also includes Metal capability lines.
    text = _run_profiler("SPMetalDataType")
    if not text:
        text = _run_profiler("SPDisplaysDataType")

    if not text:
        RESULT["notes"].append("system_profiler returned no data.")
        return False

    RESULT["details"]["raw_sample"] = "\n".join(text.splitlines()[:50])  # first 50 lines for context

    # Heuristics: look for "Metal: Supported" or "Metal Family" lines.
    patterns = [
        r"\bMetal:\s*Supported\b",
        r"\bMetal\s+Support:\s*Supported\b",
        r"\bMetal\s+Family\b",
        r"\bmacOS\s+GPUFamily",   # newer family strings
    ]
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def main():
    if not RESULT["is_macos"]:
        RESULT["notes"].append("This is not macOS (Metal is macOS-only).")
        print_and_exit(False)

    # Prefer the proper API
    if try_pyobjc():
        print_and_exit(True)

    # Fall back to system_profiler
    ok = try_system_profiler()
    print_and_exit(ok)

if __name__ == "__main__":
    main()
