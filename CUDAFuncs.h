#ifndef CUDAFUNCS_H_INCLUDED
#define CUDAFUNCS_H_INCLUDED

#include <Eigen/Dense>
#include <iostream>
#include <thrust/complex.h>
#include <omp.h>
#include <vector>

class CUDAOffloader{

    public:

        CUDAOffloader();
        void sendUToGPU(Eigen::MatrixXcd& U);
        void queryGPUDevices();
        void setGPUDevice(int deviceNumb);
        void allocateResources();
        void sendFactorialToGPU(std::vector<double>& factorial);
        void initializeStartingNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);

        int numberOfTerms;

    private:

        int blocksPerGrid, threadsPerBlock, numberOfThreads, termIntervals;

};

#endif // CUDAFUNCS_H_INCLUDED
