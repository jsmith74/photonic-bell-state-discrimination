#include "CUDAFuncs.h"

#define ANCILLA_PHOTONS 6
#define ANCILLA_MODES 8
#define HILBERT_SPACE_DIMENSION 75582

#define TERMS_BUFFER 0

__constant__ double dev_factorial[ ANCILLA_PHOTONS + 2 + 1 ];
__constant__ double dev_U[ 2 * (ANCILLA_MODES + 4) * (ANCILLA_MODES + 4) ];
thrust::complex<double>* dev_UTerms;


void CUDAOffloader::initializeStartingNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime){



    return;

}

void CUDAOffloader::allocateResources(){

    int count;

    cudaGetDeviceCount( &count );

    assert( count > 0 );

    cudaDeviceProp prop;

    cudaGetDeviceProperties( &prop,0 );

    int spaceAvail = prop.totalGlobalMem;

    int UStorageSize = ( ANCILLA_MODES + 4 ) * ( ANCILLA_MODES + 4 ) * 16;
    int factorialStorageSize = ( ANCILLA_PHOTONS + 3 ) * 8;
    int UTermStorageSize = 16 * HILBERT_SPACE_DIMENSION;

    spaceAvail -= UStorageSize;
    spaceAvail -= factorialStorageSize;
    spaceAvail -= UTermStorageSize;

    std::cout << "Space available on GPU: " << spaceAvail << " bytes" <<std::endl << std::endl;

    numberOfThreads = 0;
    int spaceTaken = 0;

    while( spaceTaken < spaceAvail ){

        numberOfThreads++;

        spaceTaken = 4 * numberOfThreads * ( 2 + 4 + ANCILLA_PHOTONS + ANCILLA_MODES );

        spaceTaken += 16 * numberOfThreads;

    }

    numberOfThreads--;

    threadsPerBlock = 1024;

    std::cout << "Number of total terms: " << numberOfTerms << std::endl;
    std::cout << "Max Number of threads: " << numberOfThreads << std::endl;
    std::cout << "Space used on GPU for this number: " << 4 * numberOfThreads * ( 2 + 4 + ANCILLA_PHOTONS + ANCILLA_MODES + 4 ) << " bytes" << std::endl;

    termIntervals = ( numberOfTerms + numberOfThreads - 1 ) / numberOfThreads;
    termIntervals += TERMS_BUFFER;

    std::cout << "The Minimum Number of Terms that need to be evaluated in at least one interval: " << termIntervals << std::endl;

    numberOfThreads = ( numberOfTerms + termIntervals - 1 ) / termIntervals;

    std::cout << "Adjusted number of threads: " << numberOfThreads << std::endl;
    std::cout << "Number of total terms covered if each thread does " << termIntervals << " terms: "  << numberOfThreads * termIntervals << std::endl;

    while( numberOfThreads % threadsPerBlock != 0 ) threadsPerBlock--;

    blocksPerGrid = numberOfThreads / threadsPerBlock;

    std::cout << "Adjusted threads per block: " << threadsPerBlock << std::endl;
    std::cout << "Adjusted blocks per grid: " << blocksPerGrid << std::endl;

    std::cout << "Adjusted Space used on GPU: " << 4 * numberOfThreads * ( 2 + 4 + ANCILLA_PHOTONS + ANCILLA_MODES + 4 ) << " bytes" << std::endl;

    assert( threadsPerBlock * blocksPerGrid == numberOfThreads );

    return;

}

void CUDAOffloader::sendUToGPU(Eigen::MatrixXcd& U){

    double UArr[ 2 * (ANCILLA_MODES + 4) * (ANCILLA_MODES + 4) ];

    int k=0;

    for(int j=0;j<ANCILLA_MODES+4;j++) for(int i=0;i<ANCILLA_MODES+4;i++){

        UArr[k] = std::real( U(i,j) );
        k++;

        UArr[k] = std::imag( U(i,j) );
        k++;

    }

    cudaMemcpyToSymbol( dev_U,UArr, 2 * (ANCILLA_MODES + 4) * (ANCILLA_MODES + 4) * sizeof(double) );

    return;

}

CUDAOffloader::CUDAOffloader(){


}


void CUDAOffloader::setGPUDevice(int deviceNumb){

    cudaSetDevice(deviceNumb);

    return;

}

void CUDAOffloader::sendFactorialToGPU(std::vector<double>& factorial){

    assert( ANCILLA_PHOTONS + 2 + 1 == factorial.size() );

    double factorialArr[factorial.size()];

    for(int i=0;i<factorial.size();i++) factorialArr[i] = factorial.at(i);

    cudaMemcpyToSymbol( dev_factorial,factorialArr, factorial.size() * sizeof(double) );

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

        std::cout << "\t" << "Compute mode: " << prop.computeMode << std::endl;
        std::cout << "\t" << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;

        prop.computeMode = 1;

        std::cout << std::endl;

    }

    return;

}
