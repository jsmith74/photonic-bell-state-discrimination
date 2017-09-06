#ifndef CUDAFUNCS_H_INCLUDED
#define CUDAFUNCS_H_INCLUDED

#include <Eigen/Dense>
#include <iostream>


class CUDAOffloader{

    public:

        CUDAOffloader();
        void sendUtoGPU(Eigen::MatrixXcd& U);
        double setMutualEntropy();
        void queryGPUDevices();
        void setGPUDevice(int deviceNumb);

        int numberOfTerms;

    private:


};

#endif // CUDAFUNCS_H_INCLUDED
