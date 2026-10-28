# Version deviations from the repo's pinned environment

Policy: get a working proof of concept first. Where an old version blocks it, try a newer one and record it here
so it can be reverted and reported later.

Working environment: conda env `neucon117`. Run the demo with `~/miniconda3/envs/neucon117/bin/python demo.py --cfg ./config/demo.yaml`.

| # | Item | Repo pin | Used | Why | Status |
|---|------|----------|------|-----|--------|
| 1 | mkl | unpinned (solver picks 2026.1) | 2024.0 | PyTorch 1.6 failed to import: `undefined symbol: iJIT_IsProfilingActive` | works in env `neucon` and `neucon117` |
| 2 | PyTorch / CUDA | 1.6.0 / 10.2 | 1.13.1 / 11.7 (env `neucon117`) | conda has no CUDA 10.2 `nvcc`. The system `nvcc` is 12.0 with gcc 13, so `torchsparse` could not be built against torch 1.6. | works: demo runs on the T400 |
| 3 | Compiler for building | system gcc 13 | conda `gxx_linux-64` 11 | CUDA 11.7 `nvcc` supports gcc up to 11 | works |
| 4 | torchsparse | git main (now a different API) | v1.4.0, built from source for sm_75 | the repo uses the 1.x API | works |
| 5 | Python | 3.7.9 | 3.7.12 | conda-forge build | works |
| 6 | pyglet | unpinned | `<2` | pyglet 2.x needs Python 3.8+; `utils.py` imports `pyrender` at module level | works |
| 7 | Evaluation-only deps (`ray`, `open3d`, `pycuda`) | requirements.txt | not installed | not used by `demo.py` | skipped |

## Build workarounds for torchsparse on Ubuntu 24.04 (not version changes)

- `libxcrypt` installed into `neucon117`: Python 3.7's `Python.h` needs `crypt.h`, which the conda gcc sysroot lacks.
- Link step failed with `cannot find /lib64/libm.so.6`: Python's build config adds `-Wl,--sysroot=/`. Fixed by building with
  `LDSHARED="$CXX -pthread -shared -L$E/lib -Wl,-rpath=$E/lib"`.
- `~/miniconda3/envs/neucon117/compiler_compat/ld` was renamed to `ld.bak` while debugging the link error. It did not fix
  the problem on its own. To revert: `mv compiler_compat/ld.bak compiler_compat/ld`.
- Build env vars used: `CUDA_HOME=$E CC/CXX=conda gcc/g++ 11 CPLUS_INCLUDE_PATH=$E/include TORCH_CUDA_ARCH_LIST=7.5`.

## Other repo changes

- `config/demo.yaml`: `TEST.PATH` points at the phone capture.
- New file `tools/process_arcore_data.py`: converts ArCoreRecorder captures to the layout `demo.py` expects.

## Revert point

Env `neucon` (PyTorch 1.6.0 + CUDA 10.2, mkl 2024.0) is kept. It runs on the GPU but cannot build `torchsparse`.

## Result of the proof of concept

Capture `rec_20261028_110419` (111 frames, 11 s, 4.6 m camera path) gave 45 keyframes and 5 fragments.
`demo.py` ran in about 4 s at roughly 10.7 keyframes/sec using 0.2 GB of GPU memory. It produced a mesh with 3,326 vertices and
6,440 faces in `results/scene_demo_checkpoints_fusion_eval_47/`. One fragment logged "no valid points: scale 1".
