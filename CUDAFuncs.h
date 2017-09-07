#ifndef CUDAFUNCS_H_INCLUDED
#define CUDAFUNCS_H_INCLUDED

#include <Eigen/Dense>
#include <iostream>
#include <thrust/complex.h>
//#include <omp.h>
#include <vector>

class CUDAOffloader{

    public:

        CUDAOffloader();
        void sendUToGPU(Eigen::MatrixXcd& U);
        double setMutualEntropy(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);
        void queryGPUDevices();
        void setGPUDevice(int deviceNumb);
        void allocateResources();
        void sendFactorialToGPU(std::vector<double>& factorial);
        int numberOfTerms;

    private:

        std::vector<int> threadsPerBlock;
        std::vector<int> blocksPerGrid;
        std::vector<int> termsPerIteration;
        int totalTermsPerIteration, iterations, numbGPUs, subIndex;

        int* nPrimeSub;
        int* mPrimeSub;

        void setSubNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);

};

#endif // CUDAFUNCS_H_INCLUDED
