#include "UGeneration.h"

#define KK 50

UGeneration::UGeneration(){



}

void UGeneration::setUCondition1(Eigen::VectorXd& position,Eigen::MatrixXcd& U){

    int k=0;

    for(int j=0;j<U.cols();j++) for(int i=0;i<U.rows();i++){

        if(zeroEntries(i,j)!=0){

            U(i,j) = position(k);
            k++;

        }

    }

    std::complex<double> I(0.0,1.0);

    for(int j=0;j<U.cols();j++) for(int i=0;i<U.rows();i++){

        if(zeroEntries(i,j)!=0){

            U(i,j) *= std::exp( I * position(k) );
            k++;

        }

    }

    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(U);

    double normalizeFactor = 0;

    for(int i=0;i<svd.singularValues().size();i++) normalizeFactor += std::pow( svd.singularValues()(i),KK );

    normalizeFactor = std::pow( normalizeFactor,1.0/KK );

    U /= normalizeFactor;

    return;

}

void UGeneration::testRandomUCondition1(Eigen::MatrixXcd& U){

    U = Eigen::MatrixXcd::Random(U.rows(),U.cols());

    U = U.array() * zeroEntriesDouble.array();

    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(U);

    double normalizeFactor = 0;

    for(int i=0;i<svd.singularValues().size();i++) normalizeFactor += std::pow(svd.singularValues()(i),KK);

    normalizeFactor = std::pow( normalizeFactor,1.0/KK );

    U /= normalizeFactor;

    return;

}



void UGeneration::initializeUCondition1(Eigen::MatrixXcd& U){

    int ancillaRows = U.rows() - 4;

    zeroEntries.resize(U.rows(),U.cols());

    initializeZeroEntries();

    setZeroEntriesRandomlyCondition1(ancillaRows);

    while( checkCondition1() ){

        initializeZeroEntries();
        setZeroEntriesRandomlyCondition1(ancillaRows);

    }

    /** =========== SET THE ZERO ENTRIES OF THE MATRIX ======================= */

//    zeroEntries << 0,1,1,1,0,1,0,0,1,1,
//                   0,1,1,1,0,1,0,0,1,1,
//                   1,0,0,0,1,0,1,1,0,0,
//                   1,0,0,0,1,0,1,1,0,0,
//                   0,1,1,1,0,1,0,0,1,1,
//                   0,1,1,1,0,1,0,0,1,1,
//                   0,1,1,1,1,1,1,0,1,1,
//                   1,1,1,1,0,1,0,1,1,1,
//                   0,1,1,1,1,1,1,0,1,1,
//                   1,1,1,1,0,1,0,1,1,1;

    /** ====================================================================== */

    std::cout << "Zero entries of the U matrix:\n" << zeroEntries << std::endl << std::endl;

    double gradientCheck = 1e-4;

    double maxStepSize = 200.0;

    BFGS_Optimization optimizer(gradientCheck,maxStepSize,zeroEntries);

    U = optimizer.minimize();

    return;

}


int UGeneration::setFuncDimension(){

    int funcDimension = 0;

    for(int i=0;i<zeroEntries.rows();i++) for(int j=0;j<zeroEntries.cols();j++){

        if(zeroEntries(i,j) != 0) funcDimension++;

    }

    funcDimension *= 2;

    return funcDimension;

}

void UGeneration::testRandomUnitary(Eigen::MatrixXcd& U){

    Eigen::MatrixXcd H = Eigen::MatrixXcd::Random( U.rows(),U.cols() );

    std::complex<double> I(0.0,1.0);

    H += H.conjugate().transpose().eval();

    H *= I;

    U = H.exp();

    return;

}

void UGeneration::testStandardUnitary(Eigen::MatrixXcd& U){

    Eigen::MatrixXcd HH( U.rows(),U.cols() );

    std::complex<double> I(0.0,1.0);

    for(int i=0;i<HH.rows();i++) for(int j=0;j<HH.cols();j++) HH(i,j) = (1.0*i) + I*(1.0*j);

    HH = HH + HH.conjugate().transpose().eval();

    HH *= I;

    U = HH.exp();

    return;

}

void UGeneration::setZeroEntriesRandomlyCondition1(int ancillaRows){

    int zeroChoiceRange = ancillaRows + 2;

    for(int j=0;j<zeroEntries.cols();j++){

        int zeroChoices[3];

        zeroChoices[0] = rand() % zeroChoiceRange;

        zeroChoices[1] = rand() % zeroChoiceRange;

        while(zeroChoices[1] == zeroChoices[0]) zeroChoices[1] = rand() % zeroChoiceRange;

        zeroChoices[2] = rand() % zeroChoiceRange;

        while(zeroChoices[2] == zeroChoices[1] || zeroChoices[2] == zeroChoices[0]) zeroChoices[2] = rand() % zeroChoiceRange;

        for(int i=0;i<3;i++){

            if(zeroChoices[i] < ancillaRows) zeroEntries(zeroChoices[i],j) = 0;
            else if(zeroChoices[i]==ancillaRows){
                zeroEntries(ancillaRows,j) = 0;
                zeroEntries(ancillaRows+1,j) = 0;
            }
            else if(zeroChoices[i]==ancillaRows+1){
                zeroEntries(ancillaRows+2,j) = 0;
                zeroEntries(ancillaRows+3,j) = 0;
            }

        }
    }

}

void UGeneration::initializeZeroEntries(){

    for(int i=0;i<zeroEntries.rows();i++) for(int j=0;j<zeroEntries.cols();j++) zeroEntries(i,j) = 1;

    return;

}

bool UGeneration::checkCondition1(){

    for(int j=0;j<zeroEntries.rows();j++){

        if( zeroEntries.row(j).sum() == 0 ) return true;

    }

    for(int j=0;j<zeroEntries.cols();j++){

        Eigen::VectorXi zeroChoices(0);
        int k=0;
        for(int i=0;i<zeroEntries.rows();i++){
            if(zeroEntries(i,j) == 0){
                zeroChoices.conservativeResize( k+1 );
                zeroChoices[k] = i;
                k++;
            }
        }

        for(int i=j+1;i<zeroEntries.cols();i++){

            bool weGood = false;

            for(int l=0;l<zeroChoices.size();l++) if(zeroEntries( zeroChoices(l),i ) == 0) weGood = true;

            if(!weGood) return true;

        }

    }

    return false;

}
