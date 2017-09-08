#include "CUDAFuncs.h"

#define ANCILLA_PHOTONS 6
#define ANCILLA_MODES 8

__constant__ double dev_factorial[ ANCILLA_PHOTONS + 2 + 1 ];
__constant__ double dev_U[ 2 * (ANCILLA_MODES + 4) * (ANCILLA_MODES + 4) ];


__global__ void setEachUTerm(){


}


void CUDAOffloader::allocateResources(){

    int count;

    cudaGetDeviceCount( &count );

    cudaDeviceProp prop[count];

    for(int i=0;i<count;i++) cudaGetDeviceProperties( &prop[i],i );

    int UStorageSize = ( ANCILLA_MODES + 4 ) * ( ANCILLA_MODES + 4 ) * 16;
    int factorialStorageSize = ( ANCILLA_PHOTONS + 3 ) * 8;

    int spaceAvail[ count ];

    for(int i=0;i<count;i++) spaceAvail[i] = prop[i].totalGlobalMem;

    for(int i=0;i<count;i++) spaceAvail[i] -= ( UStorageSize + factorialStorageSize );

    int maxTerms[ count ];

    for(int i=0;i<count;i++){

        maxTerms[i] = 0;

        int spaceTaken = 0;

        while( spaceTaken < spaceAvail[i] ){

            spaceTaken = maxTerms[i] * 4 * ( 4 + ANCILLA_MODES );
            spaceTaken += maxTerms[i] * 4 * ( 2 + ANCILLA_PHOTONS );
            spaceTaken += maxTerms[i] * 16;

            maxTerms[i]++;

        }

        maxTerms[i]--;

    }

    maxTerms[1] = 0;

    std::cout << "Terms that can be done on GTX at once: " << maxTerms[0] << std::endl;
    std::cout << "Terms that can be done on NVS at once: " << maxTerms[1] << std::endl;
    std::cout << "Total terms: " << numberOfTerms << std::endl;

    blocksPerGrid.resize(count);
    threadsPerBlock.resize(count);
    termsPerIteration.resize(count);

    for(int i=0;i<count;i++) threadsPerBlock.at(i) = prop[i].maxThreadsPerBlock;

    for(int i=0;i<count;i++) blocksPerGrid.at(i) = maxTerms[i] / threadsPerBlock[i];

    for(int i=0;i<count;i++) termsPerIteration.at(i) = blocksPerGrid.at(i) * threadsPerBlock.at(i);

    for(int i=0;i<count;i++) assert( termsPerIteration.at(i) <= maxTerms[i] );

    totalTermsPerIteration = 0;
    for(int i=0;i<count;i++) totalTermsPerIteration += termsPerIteration.at(i);

    iterations = ( numberOfTerms + totalTermsPerIteration - 1 ) / totalTermsPerIteration;

    for(int i=0;i<count;i++) std::cout << "Blocks Per Grid: " << blocksPerGrid.at(i) << std::endl;
    for(int i=0;i<count;i++) std::cout << "Threads Per Block: " << threadsPerBlock.at(i) << std::endl;
    for(int i=0;i<count;i++) std::cout << "Terms Per Iteration: " << termsPerIteration.at(i) << std::endl;

    std::cout << "Total Terms Per Iteration: " << totalTermsPerIteration << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    numbGPUs = count;

    nPrimeSub = new int[ totalTermsPerIteration * ( 4 + ANCILLA_MODES ) ];
    mPrimeSub = new int[ totalTermsPerIteration * ( 2 + ANCILLA_PHOTONS ) ];

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

void CUDAOffloader::setSubNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime){

    int kn = 0;
    int km = 0;

    int subWall = 0;

    while( subWall < totalTermsPerIteration ){

        do{

            if( subWall >= totalTermsPerIteration ){

                subIndex--;
                break;

            }

            for(int i=0;i<nPrime[ subIndex ].size();i++){

                nPrimeSub[ kn ] = nPrime[ subIndex ][i];

                kn++;

            }

            for(int i=0;i<mPrime[ subIndex ].size();i++){

                mPrimeSub[ km ] = mPrime[ subIndex ][i];

                km++;

            }

            subWall++;

        } while( std::next_permutation( mPrime[ subIndex ].begin(), mPrime[ subIndex ].end()  ) );

        subIndex++;

    }

    return;

}


double CUDAOffloader::setMutualEntropy(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime){

    subIndex = 0;

    setSubNPrimeMPrime(nPrime,mPrime);

    setEachUTerm<<<10,10>>>();

    std::cout << "CUDA Errors: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;

    return 1.0;

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
