# aCG Results Summary

Values are medians across repeats.

Input roots: `CUDA=/home/franco-merenda/Software-Projects/aCG-SYCL/temp_files/cuda_results, SYCL=/home/franco-merenda/Software-Projects/aCG-SYCL/temp_files/sycl`
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
| SYCL | 1n1g | none | 1 | 3 | 52.079 | 52.068-52.095 | 52.079 | N/A | N/A | N/A | N/A | N/A | 25185 | 141.846 | 75.260 | N/A | N/A |
| SYCL | 1n4g | mpi | 4 | 3 | 17.628 | 17.617-18.051 | 17.511 | 1.652 | 32.822 | 50340 | 0.504 | N/A | 25169 | 588.670 | 42.780 | 9.373 | 2.862 |
| SYCL | 2n4g | mpi | 8 | 3 | 12.201 | 12.190-12.580 | 12.145 | 1.840 | 36.121 | 50936 | 0.535 | N/A | 25467 | 860.569 | 36.470 | 15.079 | 4.384 |
| SYCL | 4n4g | mpi | 16 | 3 | 9.130 | 9.117-9.512 | 9.066 | N/A | N/A | N/A | N/A | N/A | 25740 | 1162.338 | 32.910 | N/A | N/A |

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
| SYCL | 1n1g | none | 1 | 3/3 | 25185 | 4.658e+08 | 9.932e-07 | 9.932e-07 | 0.145 | 52.079 | 2.068 | 141.846 |
| SYCL | 1n4g | mpi | 4 | 3/3 | 25169 | 4.682e+08 | 9.983e-07 | 9.983e-07 | 0.145 | 17.628 | 0.700 | 588.670 |
| SYCL | 2n4g | mpi | 8 | 3/3 | 25467 | 4.442e+08 | 9.471e-07 | 9.471e-07 | 0.144 | 12.201 | 0.479 | 860.569 |
| SYCL | 4n4g | mpi | 16 | 3/3 | 25740 | 4.561e+08 | 9.727e-07 | 9.727e-07 | 0.144 | 9.130 | 0.355 | 1162.338 |

## Best Solver Time By Scale

| Dataset | Scale | Best comm | Ranks | Solver s | Running s | Iters | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA | 1n1g | none | 1 | 38.981 | 40.135 | 25634 | 271.140 |
| CUDA | 1n4g | mpi | 4 | 12.588 | 14.672 | 25765 | 844.794 |
| CUDA | 2n4g | nccl | 8 | 7.846 | 9.951 | 25591 | 1348.062 |
| CUDA | 4n4g | nccl | 16 | 6.077 | 8.187 | 25358 | 1722.558 |
| SYCL | 1n1g | none | 1 | 52.079 | 52.079 | 25185 | 141.846 |
| SYCL | 1n4g | mpi | 4 | 17.628 | 17.511 | 25169 | 588.670 |
| SYCL | 2n4g | mpi | 8 | 12.201 | 12.145 | 25467 | 860.569 |
| SYCL | 4n4g | mpi | 16 | 9.130 | 9.066 | 25740 | 1162.338 |

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
| SYCL | 1n4g | mpi | 17.628 | 68.696 | 6.835 | 9.373 | 2.862 | 5.016 | 0.519 |
| SYCL | 2n4g | mpi | 12.201 | 54.036 | 9.194 | 15.079 | 4.384 | 7.206 | 0.668 |
| SYCL | 4n4g | mpi | 9.130 | 41.098 | 11.605 | N/A | 6.683 | 8.770 | 0.873 |

## Rank Skew

| Dataset | Scale | Comm | Ranks | Operation | Skew s |
| --- | --- | --- | --- | --- | --- |
| SYCL | 1n4g | mpi | 4 | spmv | 0.928 |
| SYCL | 1n4g | mpi | 4 | allreduce | 0.695 |
| SYCL | 1n4g | mpi | 4 | p2p | 0.798 |
| SYCL | 1n4g | mpi | 4 | halo_spmv | 0.194 |
| SYCL | 1n4g | mpi | 4 | pack | 0.184 |
| SYCL | 1n4g | mpi | 4 | host_sync | 0.182 |
| SYCL | 2n4g | mpi | 8 | spmv | 1.272 |
| SYCL | 2n4g | mpi | 8 | allreduce | 0.950 |
| SYCL | 2n4g | mpi | 8 | p2p | 0.867 |
| SYCL | 2n4g | mpi | 8 | halo_spmv | 0.523 |
| SYCL | 2n4g | mpi | 8 | pack | 0.182 |
| SYCL | 2n4g | mpi | 8 | host_sync | 0.184 |
| SYCL | 4n4g | mpi | 16 | spmv | 1.081 |
| SYCL | 4n4g | mpi | 16 | allreduce | 0.909 |
| SYCL | 4n4g | mpi | 16 | p2p | 0.810 |
| SYCL | 4n4g | mpi | 16 | halo_spmv | 0.473 |
| SYCL | 4n4g | mpi | 16 | pack | 0.261 |
| SYCL | 4n4g | mpi | 16 | host_sync | 0.261 |

## Partition Imbalance

| Dataset | Scale | Comm | Ranks | Local nnz max/min | Halo nnz max/min | Ghosts max/min | Imports max/min | Exports max/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n4g | mpi | 4 | 1.097 | 2.136 | 2.118 | 2.118 | 2.152 |
| SYCL | 2n4g | mpi | 8 | 1.211 | 3.198 | 3.134 | 3.134 | 3.227 |
| SYCL | 4n4g | mpi | 16 | 1.516 | 4.748 | 4.620 | 4.620 | 4.809 |

## CUDA Reference Comparison

| Scale | SYCL comm | Ranks | SYCL solver s | SYCL iters | SYCL ms/iter | CUDA MPI s | CUDA MPI iters | CUDA MPI ms/iter | CUDA best s | Solver slowdown vs MPI | Solver slowdown vs best | Iter ratio vs MPI | ms/iter ratio vs MPI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1n1g | none | 1 | 52.079 | 25185 | 2.068 | 38.981 | 25634 | 1.521 | 38.981 | 1.336 | 1.336 | 0.982 | 1.360 |
| 1n4g | mpi | 4 | 17.628 | 25169 | 0.700 | 12.588 | 25765 | 0.489 | 12.588 | 1.400 | 1.400 | 0.977 | 1.434 |
| 2n4g | mpi | 8 | 12.201 | 25467 | 0.479 | 8.436 | 25683 | 0.328 | 7.846 | 1.446 | 1.555 | 0.992 | 1.459 |
| 4n4g | mpi | 16 | 9.130 | 25740 | 0.355 | 7.803 | 25594 | 0.305 | 6.077 | 1.170 | 1.502 | 1.006 | 1.163 |

## Cross-Dataset Comparison

| Scale | Comm | Ranks | Baseline | Baseline solver s | Compared | Compared solver s | Baseline/Compared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1n1g | none | 1 | CUDA | 38.981 | SYCL | 52.079 | 0.748 |
| 1n4g | mpi | 4 | CUDA | 12.588 | SYCL | 17.628 | 0.714 |
| 2n4g | mpi | 8 | CUDA | 8.436 | SYCL | 12.201 | 0.691 |
| 4n4g | mpi | 16 | CUDA | 7.803 | SYCL | 9.130 | 0.855 |
