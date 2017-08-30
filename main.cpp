#include "LinearOpticalTransform.h"

#include <unsupported/Eigen/MatrixFunctions>

int main(){

    // TO DO: review SVD Non-unitary matrix generators and write it. Don't _always_ re-normalize the singular values
    // just do it when when the matrix becomes super unitary. We want to allow the singular values to be <=1 and
    // re-normalizing in this regime will force at least one singular value to always be 1.

    // Think about the optimization, think about an appropriate starting regime for the optimization.
    // Parallelize this, also think about implementing CUDA again with this code if timing becomes an issue.

    LinearOpticalTransform LOCircuit;

    int ancillaPhotons = 2;
    int ancillaModes = 2;

    LOCircuit.initializeCircuit(ancillaPhotons,ancillaModes);

    Eigen::MatrixXcd U,H;

    H = Eigen::MatrixXcd::Random(4+ancillaModes,4+ancillaModes);

    H += H.conjugate().transpose().eval();

    std::cout << "H:\n" << H << std::endl << std::endl;

    std::complex<double> I(0.0,1.0);

    U = ( I * H ).exp();

    std::cout << "U:\n" << std::endl << std::endl;

    std::cout << U << std::endl << std::endl;

    std::cout << U.conjugate().transpose() * U << std::endl << std::endl;

    std::cout << U * U.conjugate().transpose() << std::endl << std::endl;

    LOCircuit.setMutualEntropy(U);

    return 0;

}
