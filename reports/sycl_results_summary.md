# SYCL - Results v0.0.1 -- aCG Results Summary

Matrices: `Bump_2911`

## Solver And Communication Summary

| Dataset | Scale | Comm | Ranks | Repeats | Solver s | Solver min-max s | Running s | Allreduce s | Allreduce us/op | Allreduce calls | Halo s | Halo us/msg | Iters | GF/s | Wall s | Allreduce % | Halo % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 3 | 64.288 | 64.240-64.297 | 64.288 | N/A | N/A | N/A | N/A | N/A | 31401 | 143.269 | 86.600 | N/A | N/A |
| SYCL | 1n4g | mpi | 4 | 3 | 22.451 | 22.433-22.822 | 22.334 | 2.078 | 32.364 | 64204 | 0.639 | N/A | 32101 | 589.520 | 47.820 | 9.255 | 2.848 |
| SYCL | 2n4g | mpi | 8 | 3 | 15.427 | 15.418-15.826 | 15.377 | 2.339 | 36.241 | 64536 | 0.689 | N/A | 32267 | 862.393 | 39.910 | 15.161 | 4.467 |
| SYCL | 4n4g | mpi | 16 | 3 | 11.278 | 11.269-11.681 | 11.220 | 2.397 | 37.573 | 63796 | 0.770 | N/A | 31897 | 1166.144 | 35.310 | 21.254 | 6.829 |

## Best Solver Time By Scale

| Dataset | Scale | Best comm | Ranks | Solver s | Running s | Iters | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SYCL | 1n1g | none | 1 | 64.288 | 64.288 | 31401 | 143.269 |
| SYCL | 1n4g | mpi | 4 | 22.451 | 22.334 | 32101 | 589.520 |
| SYCL | 2n4g | mpi | 8 | 15.427 | 15.377 | 32267 | 862.393 |
| SYCL | 4n4g | mpi | 16 | 11.278 | 11.220 | 31897 | 1166.144 |
