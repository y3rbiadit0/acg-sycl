# CUDA/SYCL Comparison Results -- aCG Results Summary

Input roots: `SYCL=/home/franco-merenda/Software-Projects/aCG-SYCL/temp_files/sycl_results`
Matrices: `Bump_2911`

## Solver And Communication Summary

| Dataset | Scale | Comm | Ranks | Repeats | Solver s | Solver min-max s | Running s | Allreduce s | Allreduce us/op | Allreduce calls | Halo s | Halo us/msg | Iters | GF/s | Wall s | Allreduce % | Halo % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 3 | 64.288 | 64.240-64.297 | 64.288 | N/A | N/A | N/A | N/A | N/A | 31401 | 143.269 | 86.600 | N/A | N/A |
| SYCL | 1n4g | mpi | 4 | 3 | 22.451 | 22.433-22.822 | 22.334 | 2.078 | 32.364 | 64204 | 0.639 | N/A | 32101 | 589.520 | 47.820 | 9.255 | 2.848 |
| SYCL | 2n4g | mpi | 8 | 3 | 15.427 | 15.418-15.826 | 15.377 | 2.339 | 36.241 | 64536 | 0.689 | N/A | 32267 | 862.393 | 39.910 | 15.161 | 4.467 |
| SYCL | 4n4g | mpi | 16 | 3 | 11.278 | 11.269-11.681 | 11.220 | 2.397 | 37.573 | 63796 | 0.770 | N/A | 31897 | 1166.144 | 35.310 | 21.254 | 6.829 |

## Correctness And Runtime

| Dataset | Scale | Comm | Ranks | Converged | Iters | Residual | Rel residual r0 | Rel residual rhs | Error norm | Solver s | ms/iter | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 3/3 | 31401 | 2.439e+08 | 9.379e-07 | 9.379e-07 | 0.144 | 64.288 | 2.047 | 143.269 |
| SYCL | 1n4g | mpi | 4 | 3/3 | 32101 | 2.526e+08 | 9.714e-07 | 9.714e-07 | 0.144 | 22.451 | 0.699 | 589.520 |
| SYCL | 2n4g | mpi | 8 | 3/3 | 32267 | 2.582e+08 | 9.929e-07 | 9.929e-07 | 0.144 | 15.427 | 0.478 | 862.393 |
| SYCL | 4n4g | mpi | 16 | 3/3 | 31897 | 2.597e+08 | 9.987e-07 | 9.987e-07 | 0.144 | 11.278 | 0.354 | 1166.144 |

## Best Solver Time By Scale

| Dataset | Scale | Best comm | Ranks | Solver s | Running s | Iters | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 64.288 | 64.288 | 31401 | 143.269 |
| SYCL | 1n4g | mpi | 4 | 22.451 | 22.334 | 32101 | 589.520 |
| SYCL | 2n4g | mpi | 8 | 15.427 | 15.377 | 32267 | 862.393 |
| SYCL | 4n4g | mpi | 16 | 11.278 | 11.220 | 31897 | 1166.144 |

## Operation Breakdown

| Dataset | Scale | Comm | Solver s | SpMV/GEMV % | Dot+nrm2 % | Allreduce % | Halo/P2P % | Pack+host sync % | Other % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n4g | mpi | 22.451 | 68.753 | 6.812 | 9.255 | 2.848 | 5.009 | 0.463 |
| SYCL | 2n4g | mpi | 15.427 | 53.977 | 9.250 | 15.161 | 4.467 | 7.167 | 0.564 |
| SYCL | 4n4g | mpi | 11.278 | 41.060 | 11.607 | 21.254 | 6.829 | 8.708 | 0.767 |

## Rank Skew

| Dataset | Scale | Comm | Ranks | Operation | Skew s |
| --- | --- | --- | --- | --- | --- |
| SYCL | 1n4g | mpi | 4 | spmv | 1.177 |
| SYCL | 1n4g | mpi | 4 | allreduce | 0.885 |
| SYCL | 1n4g | mpi | 4 | p2p | 1.018 |
| SYCL | 1n4g | mpi | 4 | halo_spmv | 0.246 |
| SYCL | 1n4g | mpi | 4 | pack | 0.231 |
| SYCL | 1n4g | mpi | 4 | host_sync | 0.235 |
| SYCL | 2n4g | mpi | 8 | spmv | 1.639 |
| SYCL | 2n4g | mpi | 8 | allreduce | 1.227 |
| SYCL | 2n4g | mpi | 8 | p2p | 1.086 |
| SYCL | 2n4g | mpi | 8 | halo_spmv | 0.666 |
| SYCL | 2n4g | mpi | 8 | pack | 0.231 |
| SYCL | 2n4g | mpi | 8 | host_sync | 0.242 |
| SYCL | 4n4g | mpi | 16 | spmv | 1.325 |
| SYCL | 4n4g | mpi | 16 | allreduce | 1.087 |
| SYCL | 4n4g | mpi | 16 | p2p | 1.005 |
| SYCL | 4n4g | mpi | 16 | halo_spmv | 0.582 |
| SYCL | 4n4g | mpi | 16 | pack | 0.324 |
| SYCL | 4n4g | mpi | 16 | host_sync | 0.334 |

## Partition Imbalance

| Dataset | Scale | Comm | Ranks | Local nnz max/min | Halo nnz max/min | Ghosts max/min | Imports max/min | Exports max/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n4g | mpi | 4 | 1.097 | 2.136 | 2.118 | 2.118 | 2.152 |
| SYCL | 2n4g | mpi | 8 | 1.211 | 3.198 | 3.134 | 3.134 | 3.227 |
| SYCL | 4n4g | mpi | 16 | 1.516 | 4.748 | 4.620 | 4.620 | 4.809 |

## CUDA Reference Comparison

| Scale | SYCL comm | Ranks | SYCL solver s | SYCL iters | SYCL ms/iter | CUDA MPI s | CUDA MPI iters | CUDA MPI ms/iter | CUDA best s | Solver slowdown vs MPI | Solver slowdown vs best | Iter ratio vs MPI | ms/iter ratio vs MPI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1n1g | none | 1 | 64.288 | 31401 | 2.047 | 38.981 | 25634 | 1.521 | 38.981 | 1.649 | 1.649 | 1.225 | 1.346 |
| 1n4g | mpi | 4 | 22.451 | 32101 | 0.699 | 12.588 | 25765 | 0.489 | 12.588 | 1.784 | 1.784 | 1.246 | 1.432 |
| 2n4g | mpi | 8 | 15.427 | 32267 | 0.478 | 8.436 | 25683 | 0.328 | 7.846 | 1.829 | 1.966 | 1.256 | 1.456 |
| 4n4g | mpi | 16 | 11.278 | 31897 | 0.354 | 7.803 | 25594 | 0.305 | 6.077 | 1.445 | 1.856 | 1.246 | 1.160 |
