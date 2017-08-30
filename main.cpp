#include "LinearOpticalTransform.h"

#include <unsupported/Eigen/MatrixFunctions>

int main(){

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
