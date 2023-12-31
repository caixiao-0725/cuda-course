#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Timer.h"
// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}


__global__ void  reduceNeighbored (int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if ((tid % (2 * stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(int* g_idata, int* g_odata,
    unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling16(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 16;

    // unrolling 8
    if (idx + 15 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        int c1 = g_idata[idx + 8 * blockDim.x];
        int c2 = g_idata[idx + 9 * blockDim.x];
        int c3 = g_idata[idx + 10 * blockDim.x];
        int c4 = g_idata[idx + 11 * blockDim.x];
        int d1 = g_idata[idx + 12 * blockDim.x];
        int d2 = g_idata[idx + 13 * blockDim.x];
        int d3 = g_idata[idx + 14 * blockDim.x];
        int d4 = g_idata[idx + 15 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 + c1 + c2 + c3 + c4 + d1 + d2 + d3 + d4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int* g_idata, int* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceCompleteUnrollWarps8(int* g_idata, int* g_odata,
    unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int* g_idata, int* g_odata,
    unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(){
    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);
    
    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    Timer timer;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

    // cpu reduction
    timer.Start();
    int cpu_sum = recursiveReduce (tmp, size);
    timer.Stop();
    printf("cpu reduce elapsed %f sec cpu_sum: %d\n", timer.GetDeltaTime(), cpu_sum);
    
    // kernel 1: reduceNeighbored 邻域并行
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceNeighbored << <grid, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
	cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
	printf("gpu Neighbored elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x, block.x);


    // kernel 2: reduceNeighbored with less divergence 邻域并行，减少分支
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
    timer.Start();
	reduceNeighbored << <grid, block >> > (d_idata, d_odata, size);
	cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x, block.x);



    // kernel 3: reduceInterleaved  间域并行
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceInterleaved << <grid, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x, block.x);


    // kernel 4: reduceUnrolling2 间域并行，循环展开
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceUnrolling2 <<<grid.x / 2, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x / 2, block.x);



    // kernel 5: reduceUnrolling4 间域并行，循环展开
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceUnrolling4 << <grid.x / 4, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x / 4, block.x);


    // kernel 6: reduceUnrolling8 间域并行，循环展开
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceUnrolling8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x / 8, block.x);


    // kernel 7: reduceUnrolling16 间域并行，循环展开
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceUnrolling16 << <grid.x / 16, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x / 16 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling16  elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x / 16, block.x);


    // kernel 8: reduceUnrollWarps8 间域并行，循环展开warp
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceUnrollWarps8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x / 8, block.x);

    // kernel 9: reduceCompleteUnrollWarsp8 间域并行，几乎展开了所有的循环
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();
    reduceCompleteUnrollWarps8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu CompleteUnroll8 elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x / 8, block.x);


    // kernel 10: reduceCompleteUnroll
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.Start();

    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024> << <grid.x / 8, block >> > (d_idata, d_odata,
            size);
        break;

    case 512:
        reduceCompleteUnroll<512> << <grid.x / 8, block >> > (d_idata, d_odata,
            size);
        break;

    case 256:
        reduceCompleteUnroll<256> << <grid.x / 8, block >> > (d_idata, d_odata,
            size);
        break;

    case 128:
        reduceCompleteUnroll<128> << <grid.x / 8, block >> > (d_idata, d_odata,
            size);
        break;

    case 64:
        reduceCompleteUnroll<64> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    }

    cudaDeviceSynchronize();
    timer.Stop();
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),cudaMemcpyDeviceToHost);

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
        "%d>>>\n", timer.GetDeltaTime(), gpu_sum, grid.x / 8, block.x);


	// free host memory
	free(h_idata);
	free(h_odata);

	// free device memory
	cudaFree(d_idata);
	cudaFree(d_odata);

	// reset device
	cudaDeviceReset();

	return 0;
}
    