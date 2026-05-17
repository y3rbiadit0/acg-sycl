# CUDA Results -- aCG Results Summary

Deterministic summary generated from verbose benchmark stderr logs. Values are medians across repeats unless stated otherwise.

Input roots: `CUDA=/home/franco-merenda/Software-Projects/aCG-SYCL/temp_files/cuda_results`
Matrices: `Bump_2911`

## Solver And Communication Summary

| Dataset | Scale | Comm | Ranks | Repeats | Solver s | Solver min-max s | Running s | Allreduce s | Allreduce us/op | Allreduce calls | Halo s | Halo us/msg | Iters | GF/s | Wall s | Allreduce % | Halo % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA | 1n1g | none | 1 | 3 | 38.981 | 38.977-38.983 | 40.135 | 0 | 0 | 0 | 0 | 0 | 25634 | 271.140 | 55.770 | 0 | 0 |
| CUDA | 1n4g | mpi | 4 | 3 | 12.588 | 12.535-12.697 | 14.672 | 1.374 | 26.577 | 51532 | 0.077 | 0.062 | 25765 | 844.794 | 37.370 | 10.917 | 0.610 |
| CUDA | 1n4g | nccl | 4 | 3 | 12.601 | 11.946-12.606 | 14.722 | 0.825 | 16.055 | 52026 | 0.070 | 0.056 | 26012 | 850.833 | 38.500 | 6.549 | 0.556 |
| CUDA | 1n4g | nvshmem | 4 | 3 | 13.668 | 13.438-13.808 | 13.685 | N/A | N/A | N/A | N/A | N/A | 26003 | N/A | 36.720 | N/A | N/A |
| CUDA | 2n4g | mpi | 8 | 3 | 8.436 | 8.396-8.457 | 10.532 | 1.536 | 29.896 | 51368 | 0.072 | 0.007 | 25683 | 1255.771 | 36.060 | 18.205 | 0.849 |
| CUDA | 2n4g | nccl | 8 | 3 | 7.846 | 7.710-8.016 | 9.951 | 1.193 | 23.090 | 51184 | 0.069 | 0.007 | 25591 | 1348.062 | 36.870 | 15.202 | 0.877 |
| CUDA | 2n4g | nvshmem | 8 | 3 | 9.057 | 9.016-9.091 | 9.072 | N/A | N/A | N/A | N/A | N/A | 26509 | N/A | 34.640 | N/A | N/A |
| CUDA | 4n4g | mpi | 16 | 3 | 7.803 | 7.705-7.841 | 9.933 | 2.371 | 46.501 | 51190 | 0.684 | 0.012 | 25594 | 1353.915 | 38.300 | 30.390 | 8.768 |
| CUDA | 4n4g | nccl | 16 | 3 | 6.077 | 6.047-6.204 | 8.187 | 1.633 | 32.014 | 50718 | 0.065 | 0.001 | 25358 | 1722.558 | 38.010 | 26.874 | 1.074 |
| CUDA | 4n4g | nvshmem | 16 | 3 | 7.042 | 6.944-7.049 | 7.055 | N/A | N/A | N/A | N/A | N/A | 26664 | N/A | 35.160 | N/A | N/A |

## Correctness And Runtime

| Dataset | Scale | Comm | Ranks | Converged | Iters | Residual | Rel residual r0 | Rel residual rhs | Error norm | Solver s | ms/iter | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA | 1n1g | none | 1 | 0/3 | 25634 | 4.446e+08 | N/A | N/A | 0.144 | 38.981 | 1.521 | 271.140 |
| CUDA | 1n4g | mpi | 4 | 0/3 | 25765 | 4.343e+08 | N/A | N/A | 0.144 | 12.588 | 0.489 | 844.794 |
| CUDA | 1n4g | nccl | 4 | 0/3 | 26012 | 4.350e+08 | N/A | N/A | 0.144 | 12.601 | 0.484 | 850.833 |
| CUDA | 1n4g | nvshmem | 4 | 0/3 | 26003 | 4.434e+08 | N/A | N/A | 0.144 | 13.668 | 0.526 | 0 |
| CUDA | 2n4g | mpi | 8 | 0/3 | 25683 | 4.640e+08 | N/A | N/A | 0.144 | 8.436 | 0.328 | 1255.771 |
| CUDA | 2n4g | nccl | 8 | 0/3 | 25591 | 4.547e+08 | N/A | N/A | 0.144 | 7.846 | 0.307 | 1348.062 |
| CUDA | 2n4g | nvshmem | 8 | 0/3 | 26509 | 4.575e+08 | N/A | N/A | 0.144 | 9.057 | 0.342 | 0 |
| CUDA | 4n4g | mpi | 16 | 0/3 | 25594 | 4.417e+08 | N/A | N/A | 0.144 | 7.803 | 0.305 | 1353.915 |
| CUDA | 4n4g | nccl | 16 | 0/3 | 25358 | 4.654e+08 | N/A | N/A | 0.145 | 6.077 | 0.240 | 1722.558 |
| CUDA | 4n4g | nvshmem | 16 | 0/3 | 26664 | 4.651e+08 | N/A | N/A | 0.144 | 7.042 | 0.264 | 0 |

## Best Solver Time By Scale

| Dataset | Scale | Best comm | Ranks | Solver s | Running s | Iters | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA | 1n1g | none | 1 | 38.981 | 40.135 | 25634 | 271.140 |
| CUDA | 1n4g | mpi | 4 | 12.588 | 14.672 | 25765 | 844.794 |
| CUDA | 2n4g | nccl | 8 | 7.846 | 9.951 | 25591 | 1348.062 |
| CUDA | 4n4g | nccl | 16 | 6.077 | 8.187 | 25358 | 1722.558 |

## Operation Breakdown

| Dataset | Scale | Comm | Solver s | SpMV/GEMV % | Dot+nrm2 % | Allreduce % | Halo/P2P % | Pack+host sync % | Other % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA | 1n1g | none | 38.981 | N/A | N/A | 0 | N/A | N/A | N/A |
| CUDA | 1n4g | mpi | 12.588 | N/A | N/A | 10.917 | 0.610 | N/A | N/A |
| CUDA | 1n4g | nccl | 12.601 | N/A | N/A | 6.549 | 0.556 | N/A | N/A |
| CUDA | 1n4g | nvshmem | 13.668 | N/A | N/A | 0 | N/A | N/A | N/A |
| CUDA | 2n4g | mpi | 8.436 | N/A | N/A | 18.205 | 0.849 | N/A | N/A |
| CUDA | 2n4g | nccl | 7.846 | N/A | N/A | 15.202 | 0.877 | N/A | N/A |
| CUDA | 2n4g | nvshmem | 9.057 | N/A | N/A | 0 | N/A | N/A | N/A |
| CUDA | 4n4g | mpi | 7.803 | N/A | N/A | 30.390 | 8.768 | N/A | N/A |
| CUDA | 4n4g | nccl | 6.077 | N/A | N/A | 26.874 | 1.074 | N/A | N/A |
| CUDA | 4n4g | nvshmem | 7.042 | N/A | N/A | 0 | N/A | N/A | N/A |

## Notes

NVSHMEM/device-side CUDA logs report `allreduce` and `haloexchange` counters as zero; communication is folded into `other` in the solver breakdown. Treat NVSHMEM solver time as comparable, but not its per-operation communication timing.
For SYCL comparison, use the same matrix, seed, residual tolerances, maximum iterations, rank layout, and at least three repeats. Compare MPI-to-MPI first; compare against NCCL/NVSHMEM only as CUDA-specific upper baselines unless SYCL has equivalent communication backends.
