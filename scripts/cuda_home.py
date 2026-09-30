"""
A CUDA compiler for the kernels SGLang builds when a server starts: prints a
CUDA_HOME whose nvcc is 12.9 or newer, preferring the one pip installed next to
torch.

    export CUDA_HOME="$(python scripts/cuda_home.py)"    # gpu_pipeline.sh does this
    python scripts/cuda_home.py --check                  # ...and builds a test kernel

WHY. SGLang 0.5.20 compiles kernels at run time -- its own JIT kernels,
FlashInfer's and DeepGEMM's -- with the nvcc it finds through CUDA_HOME, then
PATH, then /usr/local/cuda. torch 2.13 brings the CUDA 13 runtime but no
compiler, and a machine's own toolkit can be older: on a Modal notebook the
agent server stopped at start-up with DeepGEMM's "NVCC version must be at least
12.9". So setup installs pip's `cuda-toolkit[nvcc,cccl]`, pinned to the version
torch pins, and this script points CUDA_HOME at it. Its layout needs two links
to be what nvcc's own profile and the JIT link lines expect: lib64 (the profile
links from lib64) and libcudart.so (the lines say -lcudart).

The interpreter matters: run it with the Python that runs SGLang.
Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_VERSION = (12, 9)
TEST_KERNEL = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/array>
__global__ void twice(__nv_bfloat16* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16(__bfloat162float(x[i]) * 2.f);
}
extern "C" int launch(__nv_bfloat16* x, int n) {
  twice<<<(n + 255) / 256, 256>>>(x, n);
  return (int)cudaGetLastError();
}
"""


def nvcc_version(home: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run([str(home / "bin" / "nvcc"), "--version"], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def pip_homes() -> list[Path]:
    """The toolkits pip installed for this interpreter (nvidia/cu13, newest first)."""
    roots = os.environ.get("PASAR_PIP_NVIDIA_ROOTS")      # tests
    if roots is not None:
        dirs = [Path(r) for r in roots.split(os.pathsep) if r]
    else:
        spec = importlib.util.find_spec("nvidia")
        dirs = [Path(d) for d in (spec.submodule_search_locations or [])] if spec else []
    homes = [h for d in dirs for h in sorted(d.glob("cu[0-9]*"), reverse=True)]
    return [h for h in homes if (h / "bin" / "nvcc").exists()]


def candidates() -> list[tuple[str, Path]]:
    found = [("pip", h) for h in pip_homes()]
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            found.append((var, Path(os.environ[var])))
    nvcc = shutil.which("nvcc")
    if nvcc:
        found.append(("PATH", Path(nvcc).resolve().parent.parent))
    found.append(("default", Path("/usr/local/cuda")))
    return found


def _driver_lib() -> Path | None:
    """The NVIDIA driver's libcuda, for code that links -lcuda."""
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/nvidia/lib64",
              "/usr/lib/aarch64-linux-gnu", "/usr/lib"):
        p = Path(d) / "libcuda.so.1"
        if p.exists():
            return p
    return None


def link_layout(home: Path) -> list[str]:
    """Give a pip toolkit the lib64 and libcudart.so its users expect. Returns
    what it linked; changes nothing that exists."""
    made = []
    lib, lib64 = home / "lib", home / "lib64"
    if lib.is_dir() and not lib64.exists():
        lib64.symlink_to("lib", target_is_directory=True)
        made.append("lib64 -> lib")
    runtime = sorted(lib.glob("libcudart.so.[0-9]*")) if lib.is_dir() else []
    if runtime and not (lib / "libcudart.so").exists():
        (lib / "libcudart.so").symlink_to(runtime[-1].name)
        made.append(f"lib/libcudart.so -> {runtime[-1].name}")
    driver = _driver_lib()
    stub = lib / "stubs" / "libcuda.so"
    if driver and lib.is_dir() and not stub.exists():
        stub.parent.mkdir(exist_ok=True)
        stub.symlink_to(driver)
        made.append(f"lib/stubs/libcuda.so -> {driver}")
    return made


def choose(found: list[tuple[str, Path]]) -> tuple[str, Path, tuple[int, int]]:
    seen = []
    for source, home in found:
        v = nvcc_version(home)
        seen.append(f"{source}: {home} ({'nvcc %d.%d' % v if v else 'no nvcc'})")
        if v and v >= MIN_VERSION:
            return source, home, v
    raise SystemExit(
        "!! no CUDA compiler 12.9 or newer. SGLang builds kernels when a server starts "
        "and needs one. Install pip's, matching torch's CUDA, into the environment that "
        "runs SGLang:\n     pip install \"cuda-toolkit[nvcc,cccl]==<the cuda-toolkit version "
        "in pip freeze>\"\n   (pn.setup(), setup_node.sh and the Modal image do this.) "
        "Looked at:\n     " + "\n     ".join(seen))


def _gpu_arch() -> str:
    try:
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.split()[0]
        major, minor = cap.split(".")
        return f"sm_{major}{minor}a" if int(major) >= 9 else f"sm_{major}{minor}"
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return "sm_90a"


def check(home: Path) -> str:
    """Build a small kernel the way SGLang's JIT does: nvcc for this GPU, the
    host compiler, CUDA and CCCL headers, a shared library linked to cudart."""
    cxx = os.environ.get("CXX", "c++")
    if not shutil.which(cxx):
        raise SystemExit(f"!! no C++ compiler ({cxx}); nvcc needs one for host code. "
                         f"apt-get install -y build-essential")
    arch = _gpu_arch()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "twice.cu"
        src.write_text(TEST_KERNEL)
        cmd = [str(home / "bin" / "nvcc"), f"-arch={arch}", "-O2", "-shared", "-Xcompiler",
               "-fPIC", str(src), "-o", str(Path(tmp) / "twice.so"),
               f"-L{home / 'lib64'}", "-lcudart"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise SystemExit(f"!! the CUDA compiler at {home} cannot build a test kernel for "
                             f"{arch}:\n{(r.stderr or r.stdout)[-1500:]}")
    return arch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="also build a test kernel")
    a = ap.parse_args(argv)
    source, home, v = choose(candidates())
    made = link_layout(home) if source == "pip" else []
    note = f"CUDA compiler {v[0]}.{v[1]} ({source}) at {home}"
    if made:
        note += f"; linked {', '.join(made)}"
    if a.check:
        note += f"; a test kernel builds for {check(home)}"
    print(note, file=sys.stderr)
    print(home)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
