#include "MeritFunction.h"

#define PI 3.141592653589793

#define SUCCESS_MUT_ENT 1.6

#define AMPLITUDE_SCALING 0.3

//#define CHECK_AMPLITUDE_SCALING

void MeritFunction::setMeritFunction(int intParam){

    int ancillaPhotons = 4;
    int ancillaModes = 4;

    /** ======================================================================

            REMEMBER TO CHECK THE AMPLITUDE SCALING FOR AN APPROPRIATE STARTING RANGE FOR EACH
            CONFIGURATION OF ANCILLA RESOURCES - MAKE SURE TO TURN OFF PARALELLIZATION IN main.cpp

        ====================================================================== */

    LOCircuit.initializeCircuit(ancillaPhotons,ancillaModes);

    funcDimension = 2 * ( 4 + ancillaModes ) * ( 4 + ancillaModes );

    U.resize( 4 + ancillaModes,4 + ancillaModes );

    #ifdef CHECK_AMPLITUDE_SCALING

        checkSVDInitialConditionScaling();

    #endif // CHECK_AMPLITUDE_SCALING

    return;

}



double MeritFunction::f(Eigen::VectorXd& position){

    std::complex<double> I(0.0,1.0);

    for(int i=0;i<funcDimension/2;i++) U( i % U.rows(), i / U.rows() ) = position(i);

    for(int i=0;i<funcDimension/2;i++) U( i % U.rows(), i / U.rows() ) *= std::exp( I * position(i + funcDimension/2) );

    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(U, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if( svd.singularValues()(0) > 1 ) U /= svd.singularValues()(0);

    LOCircuit.setMutualEntropy(U);

    return LOCircuit.mutualEntropy;

}


void MeritFunction::printReport(Eigen::VectorXd& position){

    std::complex<double> I(0.0,1.0);

    for(int i=0;i<funcDimension/2;i++) U( i % U.rows(), i / U.rows() ) = position(i);

    for(int i=0;i<funcDimension/2;i++) U( i % U.rows(), i / U.rows() ) *= std::exp( I * position(i + funcDimension/2) );

    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(U, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if( svd.singularValues()(0) > 1 ) U /= svd.singularValues()(0);

    LOCircuit.setMutualEntropy(U);

    std::cout << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl << std::endl;

    std::cout << "U:\n" << std::setprecision(6) << U << std::endl << std::endl;

    std::cout << "Singular values of U (prior to normalization):\n" << std::setprecision(6) << svd.singularValues() << std::endl << std::endl;

    if(2.0 - 0.25 * LOCircuit.mutualEntropy > SUCCESS_MUT_ENT){

        std::ofstream outfile("Success.dat",std::ofstream::app);

        outfile << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl << std::endl;

        outfile << "U:\n" << std::setprecision(6) << U << std::endl << std::endl;

        outfile << "Singular values of U (prior to normalization):\n" << std::setprecision(6) << svd.singularValues() << std::endl << std::endl;

        for(int i=0;i<position.size();i++) outfile << std::setprecision(16) << position(i) << ",";

        outfile << std::endl << std::endl;

        outfile.close();

    }

    return;

}



Eigen::VectorXd MeritFunction::setInitialPosition(){

    Eigen::VectorXd position = Eigen::VectorXd::Random(funcDimension);

    for(int i=0;i<funcDimension/2;i++) position(i) *= AMPLITUDE_SCALING;

    for(int i=funcDimension/2;i<funcDimension;i++) position(i) *= PI;

    return position;

}


void MeritFunction::checkSVDInitialConditionScaling(){

    for(int i=0;i<1000;i++){

        Eigen::VectorXd position = setInitialPosition();

        std::complex<double> I(0.0,1.0);

        for(int i=0;i<funcDimension/2;i++) U( i % U.rows(), i / U.rows() ) = position(i);

        for(int i=0;i<funcDimension/2;i++) U( i % U.rows(), i / U.rows() ) *= std::exp( I * position(i + funcDimension/2) );

        std::cout << U << std::endl << std::endl;

        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(U, Eigen::ComputeThinU | Eigen::ComputeThinV);

        std::cout << svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().conjugate().transpose() << std::endl << std::endl;

        std::cout << svd.singularValues() << std::endl;

        std::ofstream outfile("SVTest.dat",std::ofstream::app);

        for(int i=0;i<svd.singularValues().size();i++) outfile << std::setprecision(16) << svd.singularValues()(i) << "\t";

        outfile << std::endl;

        outfile.close();

    }

    assert( false );

    return;

}


MeritFunction::MeritFunction(){



}
