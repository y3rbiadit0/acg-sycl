# SYCL Results v0.0.2 -- aCG Results Summary

Input roots: `SYCL=/home/franco-merenda/Software-Projects/aCG-SYCL/temp_files/sycl_results`
Matrices: `Bump_2911`

## Solver And Communication Summary

| Dataset | Scale | Comm | Ranks | Repeats | Solver s | Solver min-max s | Running s | Allreduce s | Allreduce us/op | Allreduce calls | Halo s | Halo us/msg | Iters | GF/s | Wall s | Allreduce % | Halo % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 3 | 52.079 | 52.068-52.095 | 52.079 | N/A | N/A | N/A | N/A | N/A | 25185 | 141.846 | 75.260 | N/A | N/A |
| SYCL | 1n4g | mpi | 4 | 3 | 17.628 | 17.617-18.051 | 17.511 | 1.652 | 32.822 | 50340 | 0.504 | N/A | 25169 | 588.670 | 42.780 | 9.373 | 2.862 |
| SYCL | 2n4g | mpi | 8 | 3 | 12.201 | 12.190-12.580 | 12.145 | 1.840 | 36.121 | 50936 | 0.535 | N/A | 25467 | 860.569 | 36.470 | 15.079 | 4.384 |
| SYCL | 4n4g | mpi | 16 | 3 | 9.130 | 9.117-9.512 | 9.066 | N/A | N/A | N/A | N/A | N/A | 25740 | 1162.338 | 32.910 | N/A | N/A |

## Correctness And Runtime

| Dataset | Scale | Comm | Ranks | Converged | Iters | Residual | Rel residual r0 | Rel residual rhs | Error norm | Solver s | ms/iter | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 3/3 | 25185 | 4.658e+08 | 9.932e-07 | 9.932e-07 | 0.145 | 52.079 | 2.068 | 141.846 |
| SYCL | 1n4g | mpi | 4 | 3/3 | 25169 | 4.682e+08 | 9.983e-07 | 9.983e-07 | 0.145 | 17.628 | 0.700 | 588.670 |
| SYCL | 2n4g | mpi | 8 | 3/3 | 25467 | 4.442e+08 | 9.471e-07 | 9.471e-07 | 0.144 | 12.201 | 0.479 | 860.569 |
| SYCL | 4n4g | mpi | 16 | 3/3 | 25740 | 4.561e+08 | 9.727e-07 | 9.727e-07 | 0.144 | 9.130 | 0.355 | 1162.338 |

## Best Solver Time By Scale

| Dataset | Scale | Best comm | Ranks | Solver s | Running s | Iters | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 52.079 | 52.079 | 25185 | 141.846 |
| SYCL | 1n4g | mpi | 4 | 17.628 | 17.511 | 25169 | 588.670 |
| SYCL | 2n4g | mpi | 8 | 12.201 | 12.145 | 25467 | 860.569 |
| SYCL | 4n4g | mpi | 16 | 9.130 | 9.066 | 25740 | 1162.338 |

## Operation Breakdown

| Dataset | Scale | Comm | Solver s | SpMV/GEMV % | Dot+nrm2 % | Allreduce % | Halo/P2P % | Pack+host sync % | Other % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
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
