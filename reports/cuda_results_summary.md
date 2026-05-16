# aCG - CUDA -- Results Summary

_Values are medians across repeats._

- Binary: `acg-cuda`
- Suitesparse Matrix: `Bump_2911`

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

## Best Solver Time By Scale

| Dataset | Scale | Best comm | Ranks | Solver s | Running s | Iters | GF/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA | 1n1g | none | 1 | 38.981 | 40.135 | 25634 | 271.140 |
| CUDA | 1n4g | mpi | 4 | 12.588 | 14.672 | 25765 | 844.794 |
| CUDA | 2n4g | nccl | 8 | 7.846 | 9.951 | 25591 | 1348.062 |
| CUDA | 4n4g | nccl | 16 | 6.077 | 8.187 | 25358 | 1722.558 |