#include "CUDAFuncs.h"

#define ANCILLA_PHOTONS 6
#define ANCILLA_MODES 8



__global__ void setEachUTerm(){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int test = 33;
    int test2 = 52;

    int test3 = test + test2;

}

void CUDAOffloader::sendUtoGPU(Eigen::MatrixXcd& U){

    return;

}

double CUDAOffloader::setMutualEntropy(){

    const int threadsPerBlock = 1014;
    const int blocksPerGrid = ( numberOfTerms + threadsPerBlock - 1 ) / threadsPerBlock;;

    std::cout << blocksPerGrid << "\t" << numberOfTerms << std::endl;
    std::cout << blocksPerGrid * threadsPerBlock << std::endl;

    setEachUTerm<<<blocksPerGrid,threadsPerBlock>>>();

    return 1.0;

}


CUDAOffloader::CUDAOffloader(){


}


void CUDAOffloader::setGPUDevice(int deviceNumb){

    cudaSetDevice(deviceNumb);

    return;

}

void CUDAOffloader::queryGPUDevices(){

    int count;

    cudaGetDeviceCount( &count );

    std::cout << "Number of devices: " << count << std::endl << std::endl;

    for(int i=0;i<count;i++){

        cudaDeviceProp prop;

        cudaGetDeviceProperties( &prop,i );

        std::cout << "Device No. " << i << ": " << std::endl;
        std::cout << "\t" << prop.name << std::endl;
        if(prop.integrated) std::cout << "\tIntegrated GPU" << std::endl;
        else std::cout << "\tNon-integrated GPU" << std::endl;
        std::cout << "\t" << "Device compute capability: " << prop.major << "." << prop.minor << " (1.3 or higher supports double-precision math)" << std::endl;
        std::cout << "\t" << prop.totalGlobalMem << " bytes of global memory" << std::endl;
        std::cout << "\t" << prop.sharedMemPerBlock << " bytes of shared memory for a single block" << std::endl;
        std::cout << "\t" << prop.regsPerBlock << " registers (32 bit) available per block" << std::endl;
        std::cout << "\t" << prop.warpSize << " threads in a warp" << std::endl;
        std::cout << "\t" << prop.memPitch << " bytes maximum pitch allowed for memory copies" << std::endl;
        std::cout << "\t" << prop.maxThreadsPerBlock << " maximum number of threads that a block may contain" << std::endl;
        std::cout << "\t" << prop.maxThreadsDim[0] << " maximum number of threads along X" << std::endl;
        std::cout << "\t" << prop.totalConstMem << " amount of available constant memory" << std::endl;
        if(prop.deviceOverlap) std::cout << "\t" << "Device can simultaneously perform cudaMemcpy() and a kernel execution" << std::endl;
        else std::cout << "\t" << "Device cannot simultaneously perform cudaMemcpy() and a kernel execution" << std::endl;
        std::cout << "\t" << prop.multiProcessorCount << " multiprocessors on the device" << std::endl;
        if(prop.kernelExecTimeoutEnabled) std::cout << "\tRuntime limit for kernels on this device is enabled" << std::endl;
        else std::cout << "\tRuntime limit for kernels on this device is disabled" << std::endl;

        std::cout << std::endl;

    }

    return;

}
