#include<stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main(){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.globalL1CacheSupported) {
        printf("Global L1 cache supported\n");
    }
    else {
		printf("Global L1 cache NOT supported\n");
    }

    cudaSharedMemConfig SharedMemConfig;
    cudaDeviceGetSharedMemConfig(&SharedMemConfig);
    printf("SharedMemConfig = %d\n", SharedMemConfig);
    if(SharedMemConfig == cudaSharedMemBankSizeEightByte){
		printf("cudaSharedMemBankSizeEightByte\n");
	}
	else if(SharedMemConfig == cudaSharedMemBankSizeFourByte){
		printf("cudaSharedMemBankSizeFourByte\n");
	}
	else if(SharedMemConfig == cudaSharedMemBankSizeDefault){
		printf("cudaSharedMemBankSizeDefault\n");
	}
	else{
		printf("cudaSharedMemBankSizeInvalid\n");
	}
}