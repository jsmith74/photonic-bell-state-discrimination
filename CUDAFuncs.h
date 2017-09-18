#ifndef CUDAFUNCS_H_INCLUDED
#define CUDAFUNCS_H_INCLUDED

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <thrust/complex.h>
#include <omp.h>
#include <vector>
#include "OptimizedFunctions.h"

class CUDAOffloader{

    public:

        CUDAOffloader();
        void sendUToGPU(Eigen::MatrixXcd& U);
        void queryGPUDevices();
        void setGPUDevice(int deviceNumb);
        void allocateResources();
        void sendFactorialToGPU(std::vector<double>& factorial);
        void initializeStartingNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);
        double setMutualEntropy();
        void setReduceGrid(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);

        int numberOfTerms;

    private:

        int blocksPerGrid, threadsPerBlock, numberOfThreads, termIntervals, reduceGridSize;
        int* nPrimeStarter;
        int* mPrimeStarter;
        int* reduceGridStart;
        int* reduceGridEnd;
        int* reducePatchGrid;
        OptimizedFunctions gccCompiledFunctions;

};

#endif // CUDAFUNCS_H_INCLUDED
