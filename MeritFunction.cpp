#include "MeritFunction.h"

#define PI 3.141592653589793

#define SUCCESS_MUT_ENT 1.55

#define INITIAL_CONDITION_RANDOM_DEGREE 2000

#define AMPLITUDE_SCALING 1000.0

#define SVD_SCALING 1.0

void MeritFunction::setMeritFunction(int intParam){

    int ancillaPhotons = 4;
    int ancillaModes = 4;

    /** ======================================================================

            REMEMBER TO CHECK THE AMPLITUDE SCALING FOR AN APPROPRIATE STARTING RANGE FOR EACH
            CONFIGURATION OF ANCILLA RESOURCES - MAKE SURE TO TURN OFF PARALELLIZATION IN main.cpp

        ====================================================================== */

    LOCircuit.initializeCircuit(ancillaPhotons,ancillaModes);

    funcDimension = 2 * ( 4 + ancillaModes ) * ( 4 + ancillaModes ) + 4 + ancillaModes;

    U.resize( 4 + ancillaModes,4 + ancillaModes );
    V.resize( 4 + ancillaModes,4 + ancillaModes );
    W.resize( 4 + ancillaModes,4 + ancillaModes );

    #ifdef CHECK_AMPLITUDE_SCALING

        checkSVDInitialConditionScaling();

    #endif // CHECK_AMPLITUDE_SCALING

    return;

}



double MeritFunction::f(Eigen::VectorXd& position){

    Eigen::MatrixXcd H( U.rows(),U.cols() );

    setAntiHermitian1( H, position );

    V = H.exp();

    setAntiHermitian2( H, position );

    W = H.exp();

    for(int i=0;i<D.size();i++) D(i) = std::exp( -position( i + 2 * U.rows() * U.rows() )*position( i + 2 * U.rows() * U.rows()) );

    U = V * D.asDiagonal() * W;

    LOCircuit.setMutualEntropy(U);

    return LOCircuit.mutualEntropy;

}


void MeritFunction::printReport(Eigen::VectorXd& position){

    Eigen::MatrixXcd H( U.rows(),U.cols() );

    setAntiHermitian1( H, position );

    V = H.exp();

    setAntiHermitian2( H, position );

    W = H.exp();

    for(int i=0;i<D.size();i++) D(i) = std::exp( -position( i + 2 * U.rows() * U.rows() )*position( i + 2 * U.rows() * U.rows()) );

    U = V * D.asDiagonal() * W;

    LOCircuit.setMutualEntropy(U);

    std::cout << "Mutual Information: " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl;
    std::cout << "Total Prob Check: " << LOCircuit.totalProbCheck << std::endl;

    LOCircuit.setTotalSuccessProb(U);

    std::cout << "Success Probability: " << std::setprecision(16) << LOCircuit.successProbability << std::endl;
    std::cout << "Failure Probability: " << std::setprecision(16) << LOCircuit.failureProbability << std::endl;

    return;

}



Eigen::VectorXd MeritFunction::setInitialPosition(){

    V = Eigen::MatrixXcd::Identity(U.rows(),U.cols());
    W = Eigen::MatrixXcd::Identity(U.rows(),U.cols());

    Eigen::VectorXd position(funcDimension);

    int ampSize = U.rows() + ( U.rows() * U.rows() - U.rows() ) / 2;

    for(int j=0;j<INITIAL_CONDITION_RANDOM_DEGREE;j++){

        position = SVD_SCALING * Eigen::VectorXd::Random(funcDimension);

        for(int i=0;i<ampSize;i++) position(i) *= AMPLITUDE_SCALING;

        for(int i=0;i<ampSize;i++) position( i + U.rows() * U.rows() ) *= AMPLITUDE_SCALING;

        Eigen::MatrixXcd H( U.rows(),U.cols() );

        setAntiHermitian1( H, position );

        Eigen::MatrixXcd UTemp( H.rows(),H.cols() );
        UTemp = H.exp();

        V = UTemp * V;

        setAntiHermitian2( H, position );

        UTemp = H.exp();

        W = UTemp * W;

    }

    D.resize(U.rows());

    setPosition1( position );
    setPosition2( position );

    return position;

}

void MeritFunction::setPosition1( Eigen::VectorXd& position ){

    Eigen::MatrixXcd H(V.rows(),V.cols());

    std::complex<double> I(0.0,1.0);

    H = V.log();

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

void MeritFunction::setPosition2( Eigen::VectorXd& position ){

    Eigen::MatrixXcd H(W.rows(),W.cols());

    std::complex<double> I(0.0,1.0);

    H = W.log();

    H /= I;

    int k = H.rows() * H.rows();

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

void MeritFunction::setAntiHermitian1( Eigen::MatrixXcd& H,Eigen::VectorXd& position ){

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

void MeritFunction::setAntiHermitian2( Eigen::MatrixXcd& H,Eigen::VectorXd& position ){

    int k = H.rows() * H.rows();

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
