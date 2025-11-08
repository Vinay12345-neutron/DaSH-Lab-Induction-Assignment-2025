| Implementation             | Time (ms) | GFLOPS  | % of cuBLAS | Speedup vs Naive |
|----------------------------|-----------|---------|-------------|------------------|
| Naive                      |           |         |             | 1.0x             |
| Tiled (shared memory)      |           |         |             |                  |
| Tiled + Coalesced          |           |         |             |                  |
| + Register blocking        |           |         |             |                  |
| [Your optimizations...]    |           |         |             |                  |
| cuBLAS                     |           |         |             |                  |