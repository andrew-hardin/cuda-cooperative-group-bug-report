# Race condition reported in `cg::reduce()`
This is a minimum reproducible test case that causes the `compute-sanitizer`
to report a race condition when using the cooperative groups `cg::reduce()` method.

## Test Environment
- GPU: Quadro RTX 5000
- CUDA Compiler: nvcc V12.2.140
- Host Compiler: GCC 10 (via `devtoolset-10`)
- Operating System: RHEL7

## Steps to reproduce
```bash
# Build it.
mkdir build && cd build
cmake ..

# Run the executable under the racecheck tool.
compute-sanitizer --tool=racecheck ./demo

# Optional: run with --racecheck-report all for more output.
compute-sanitizer --tool=racecheck --racecheck-report all ./demo
```

## Expected output
```
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at 0xa60 in void Kernel<(int)256>(const int *, int, int *)
=========     and Read access at 0xcc0 in void Kernel<(int)256>(const int *, int, int *) [28 hazards]
=========
========= Error: Race reported between Write access at 0xd60 in void Kernel<(int)256>(const int *, int, int *)
=========     and Read access at 0xf30 in void Kernel<(int)256>(const int *, int, int *) [4 hazards]
=========
Host min = 10; Device min = 10
Host and device calculation match - we're good!
========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```
