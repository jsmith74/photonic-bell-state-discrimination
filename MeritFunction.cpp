#include "MeritFunction.h"

#define PI 3.141592653589793

#define SUCCESS_MUT_ENT 1.55

#define AMPLITUDE_SCALING 1000.0

#define INITIAL_CONDITION_RANDOM_DEGREE 2000

void MeritFunction::setMeritFunction(int intParam){

    int ancillaPhotons = 6;
    int ancillaModes = 8;

    LOCircuit.initializeCircuit(ancillaPhotons,ancillaModes,intParam);

    funcDimension = (4 + ancillaModes) * (4 + ancillaModes);

    U.resize( 4 + ancillaModes,4 + ancillaModes );

    return;

}



double MeritFunction::f(Eigen::VectorXd& position){

    setAntiHermitian( U , position );

    U = U.exp().eval();

    LOCircuit.setMutualEntropy(U);

    return LOCircuit.mutualEntropy;

}


void MeritFunction::printReport(Eigen::VectorXd& position){

    setAntiHermitian( U , position );

    U = U.exp().eval();

    LOCircuit.setMutualEntropy(U);

    std::ofstream outfile("resultMonitor.dat",std::ofstream::app);

    outfile << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl << std::endl;

    outfile.close();

    if(2.0 - 0.25 * LOCircuit.mutualEntropy > SUCCESS_MUT_ENT){

        outfile.open("Success.dat",std::ofstream::app);

        outfile << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl << std::endl;

        outfile << "U:\n" << std::setprecision(6) << U << std::endl << std::endl;

        for(int i=0;i<position.size();i++) outfile << std::setprecision(16) << position(i) << ",";

        outfile << std::endl << std::endl;

        outfile.close();

    }

    return;

}



Eigen::VectorXd MeritFunction::setInitialPosition(){

    U = Eigen::MatrixXcd::Identity(U.rows(),U.cols());

    Eigen::VectorXd position(funcDimension);

    int ampSize = U.rows() + ( U.rows() * U.rows() - U.rows() ) / 2;

    for(int j=0;j<INITIAL_CONDITION_RANDOM_DEGREE;j++){

        position = PI * Eigen::VectorXd::Random(funcDimension);

        for(int i=0;i<ampSize;i++) position(i) *= AMPLITUDE_SCALING;

        Eigen::MatrixXcd H( U.rows(),U.cols() );

        setAntiHermitian( H, position );

        Eigen::MatrixXcd UTemp( H.rows(),H.cols() );
        UTemp = H.exp();

        U = UTemp * U;

    }

    setPosition( U, position );

    return position;

}

void MeritFunction::setPosition(Eigen::MatrixXcd& U, Eigen::VectorXd& position){

    Eigen::MatrixXcd H(U.rows(),U.cols());

    std::complex<double> I(0.0,1.0);

    H = U.log();

    H /= I;

    int k = 0;

    for(int i=0;i<H.rows();i++) for(int j=i;j<H.cols();j++){

        position(k) = std::sqrt( std::norm(H(i,j)) ) ;
        if( i==j && std::real(H(i,j)) < 0 ) position(k) *= -1;
        k++;

    }

    for(int i=0;i<H.rows();i++) for(int j=i+1;j<H.cols();j++){

        position(k) = std::arg( H(i,j) );
        k++;

    }

    return;

}

void MeritFunction::setAntiHermitian( Eigen::MatrixXcd& H,Eigen::VectorXd& position ){

    int k = 0;

    for(int i=0;i<H.rows();i++) for(int j=i;j<H.cols();j++){

        H(i,j) = position(k);
        H(j,i) = position(k);

        k++;

    }

    std::complex<double> I(0.0,1.0);

    for(int i=0;i<H.rows();i++) for(int j=i+1;j<H.cols();j++){

        H(i,j) *= std::exp( I * position(k) );

        H(j,i) *= std::exp( -I * position(k) );

        k++;

    }

    H = I * H;

    return;

}


MeritFunction::MeritFunction(){



}
