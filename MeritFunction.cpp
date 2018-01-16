#include "MeritFunction.h"

#define PI 3.141592653589793

#define SUCCESS_MUT_ENT 1.55

#define INITIAL_CONDITION_RANDOM_DEGREE 2000

#define AMPLITUDE_SCALING 1000.0

#define SVD_SCALING 1.0

#define CONDITIONED_U

void MeritFunction::setMeritFunction(Eigen::MatrixXi& intParam){

    int ancillaPhotons = 6;
    int ancillaModes = 6;

    funcDimension = ( 4 + ancillaModes ) * ( 4 + ancillaModes );

    U.resize( 4 + ancillaModes,4 + ancillaModes );

    zeroEntries = intParam;

    return;

}



double MeritFunction::f(Eigen::VectorXd& position){

    Eigen::MatrixXcd H( V.rows(),V.cols() );

    setAntiHermitian1( H, position );

    V = H.exp();

    double output = 0.0;

    for(int j=0;j<zeroEntries.cols();j++) for(int i=0;i<zeroEntries.rows();i++){

        if(zeroEntries(i,j) == 0) output += std::norm( V(i,j) );

    }

#ifdef CONDITIONED_U

    return output;

#endif // CONDITIONED_U

    return 2.0;

}


Eigen::MatrixXcd MeritFunction::printReport(Eigen::VectorXd& position){

    Eigen::MatrixXcd H( V.rows(),V.cols() );

    setAntiHermitian1( H, position );

    V = H.exp();

    std::cout << V << std::endl << std::endl;

    std::cout << "======================================" << std::endl << std::endl;

    return V;

}



Eigen::VectorXd MeritFunction::setInitialPosition(){

    V = Eigen::MatrixXcd::Identity(U.rows(),U.cols());

    Eigen::VectorXd position(funcDimension);

    for(int j=0;j<INITIAL_CONDITION_RANDOM_DEGREE;j++){

        position = SVD_SCALING * Eigen::VectorXd::Random(funcDimension);

        for(int i=0;i<funcDimension;i++) position(i) *= AMPLITUDE_SCALING;

        Eigen::MatrixXcd H( U.rows(),U.cols() );

        setAntiHermitian1( H, position );

        Eigen::MatrixXcd UTemp( H.rows(),H.cols() );
        UTemp = H.exp();

        V = UTemp * V;

    }

    setPosition1( position );

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
