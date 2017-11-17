#include "MeritFunction.h"

#define PI 3.141592653589793

#define SUCCESS_MUT_ENT 1.55

#define INITIAL_CONDITION_RANDOM_DEGREE 2000

#define AMPLITUDE_SCALING 1000.0

#define SVD_SCALING 1.0

#define ZERO_ENTRY_WEIGHT 0.1

#define START_NEAR_ZERO_SOLUTION

void MeritFunction::setMeritFunction(int intParam){

    int ancillaPhotons = 6;
    int ancillaModes = 8;

    LOCircuit.initializeCircuit(ancillaPhotons,ancillaModes,intParam);

    funcDimension = 2 * ( 4 + ancillaModes ) * ( 4 + ancillaModes ) + ( 4 + ancillaModes );

    U.resize( 4 + ancillaModes,4 + ancillaModes );
    V.resize( 4 + ancillaModes,4 + ancillaModes );
    W.resize( 4 + ancillaModes,4 + ancillaModes );

    return;

}



double MeritFunction::f(Eigen::VectorXd& position){

    Eigen::MatrixXcd H( U.rows(),U.cols() );

    setAntiHermitian1( H, position );

    V = H.exp();

    setAntiHermitian2( H, position );

    W = H.exp();

    for(int i=0;i<D.size();i++) D(i) = std::exp( -position( i + 2 * U.rows() * U.rows() ) * position( i + 2 * U.rows() * U.rows()) );

    U = V * D.asDiagonal() * W;

    LOCircuit.setMutualEntropy(U);

    UGenerator.setZeroEntryQuant(U);

    return LOCircuit.mutualEntropy + ZERO_ENTRY_WEIGHT * UGenerator.zeroEntryQuant;

}

double MeritFunction::zeroEntryMonitor(){

     return UGenerator.zeroEntryQuant;

}

double MeritFunction::entropyMonitor(){

    return 2.0 - 0.25 * LOCircuit.mutualEntropy;

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

    std::ofstream outfile("resultMonitor.dat",std::ofstream::app);

    outfile << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl << std::endl;

    outfile.close();

    if(2.0 - 0.25 * LOCircuit.mutualEntropy > SUCCESS_MUT_ENT){

        outfile.open("Success.dat",std::ofstream::app);

        outfile << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl << std::endl;

        outfile << "U:\n" << std::setprecision(6) << U << std::endl << std::endl;

        outfile << "Zero matrix:\n" << UGenerator.zeroEntries << std::endl << std::endl;

        for(int i=0;i<position.size();i++) outfile << std::setprecision(16) << position(i) << ",";

        outfile << std::endl << std::endl;

        outfile.close();

    }

    return;

}



Eigen::VectorXd MeritFunction::setInitialPosition(){

    UGenerator.initializeUCondition1( U );

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

    #ifdef START_NEAR_ZERO_SOLUTION

        shiftUToZeroSolution( position );

    #endif // START_NEAR_ZERO_SOLUTION

    return position;

}

void MeritFunction::shiftUToZeroSolution(Eigen::VectorXd& position){

    Eigen::MatrixXcd H( U.rows(),U.cols() );

    setAntiHermitian1( H, position );

    V = H.exp();

    setAntiHermitian2( H, position );

    W = H.exp();

    for(int i=0;i<D.size();i++) D(i) = std::exp( -position( i + 2 * U.rows() * U.rows() ) * position( i + 2 * U.rows() * U.rows()) );

    U = V * D.asDiagonal() * W;

    for(int j=0;j<U.cols();j++) for(int i=0;i<U.rows();i++){

        if( UGenerator.zeroEntries(i,j) == 0 ) U(i,j) *= 0;

    }


    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(U, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if( svd.singularValues()(0) > 1 ) D = svd.singularValues() / svd.singularValues()(0);
    else D = svd.singularValues();

    V = svd.matrixU();

    W = svd.matrixV().conjugate().transpose();

    setPosition1( position );
    setPosition2( position );

    for(int i=0;i<D.size();i++) position( i + 2 * U.rows() * U.rows() ) = std::sqrt( -std::log( D(i) ) );

    return;

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
