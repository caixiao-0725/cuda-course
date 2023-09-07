#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
__global__ void scan4() {

    int laneId = threadIdx.x & 0x1f;
    int val = 8 - laneId;
    printf("laneId id: %d Thread id :%d n value = %d\n", laneId, threadIdx.x, val);
    for (int offset = 8 >> 1; offset > 0; offset >>= 1)
    {
        int n = __shfl_down_sync(0xff, val, offset, 8);
        printf("Block id: %d Thread id :%d n value = %d\n", blockIdx.x, threadIdx.x, n);
    }
}
int main() {
    scan4 << < 2, 8 >> > ();
    cudaDeviceSynchronize();
    return 0;
}

