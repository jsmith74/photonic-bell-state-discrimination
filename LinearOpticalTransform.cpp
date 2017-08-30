#include "LinearOpticalTransform.h"

LinearOpticalTransform::LinearOpticalTransform(){



}

void LinearOpticalTransform::initializeCircuit(int& ancillaP,int& ancillaM){

    ancillaPhotons = ancillaP;
    ancillaModes = ancillaM;
    HSDimension = g( ancillaPhotons + 2,ancillaModes + 4 );

    nPrime.resize( HSDimension );
    mPrime.resize( HSDimension );

    for(int i=0;i<HSDimension;i++){ nPrime.at(i).resize(ancillaModes + 4); mPrime.at(i).resize(ancillaPhotons + 2); }

    setNPrimeAndMPrime(nPrime,mPrime);

    factorial.resize( ancillaPhotons + 2 + 1 );

    for(int i=0;i<factorial.size();i++) factorial[i] = doublefactorial(i);

    assert( ancillaModes >= ancillaPhotons );

    return;

}

void LinearOpticalTransform::setNPrimeAndMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime){

    Eigen::MatrixXi outBasis;

    setToFullHilbertSpace( ancillaPhotons + 2,ancillaModes + 4,outBasis );

    for(int i=0;i<HSDimension;i++){

        for(int j=0;j<nPrime.at(i).size();j++) nPrime.at(i).at(j) = outBasis(i,j);

        setmVec(mPrime.at(i),nPrime.at(i));

    }

    return;

}

void LinearOpticalTransform::setmVec(std::vector<int>& m, std::vector<int>& n){

    int k=0;

    for(int i=0;i<n.size();i++){

        for(int j=0;j<n.at(i);j++){

            m.at(k) = i;

            k++;

        }

    }

    return;
}


template<typename T>
void LinearOpticalTransform::printVec(std::vector<T>& vec){

    for(int i=0;i<vec.size();i++) std::cout << vec.at(i) << " ";

    std::cout << std::endl;

    return;

}

void LinearOpticalTransform::setMutualEntropy(Eigen::MatrixXcd& U){

    double pyx[4];

    mutualEntropy = 0.0;

    for(int y=0;y<HSDimension;y++){

        std::complex<double> stateAmplitude[4];

        stateAmplitude[0] = 0.0;
        stateAmplitude[1] = 0.0;
        stateAmplitude[2] = 0.0;
        stateAmplitude[3] = 0.0;

        do{

            setStateAmplitude(stateAmplitude,U,y);

        } while( std::next_permutation( mPrime[y].begin(), mPrime[y].end() ) );

        normalizeStateAmplitude(stateAmplitude,y);

        pyx[0] = std::norm( stateAmplitude[0] );
        pyx[1] = std::norm( stateAmplitude[1] );
        pyx[2] = std::norm( stateAmplitude[2] );
        pyx[3] = std::norm( stateAmplitude[3] );

        if(pyx[0] != 0.0) mutualEntropy += pyx[0] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[0] );
        if(pyx[1] != 0.0) mutualEntropy += pyx[1] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[1] );
        if(pyx[2] != 0.0) mutualEntropy += pyx[2] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[2] );
        if(pyx[3] != 0.0) mutualEntropy += pyx[3] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[3] );

    }

    mutualEntropy = 2.0 - 0.25 * mutualEntropy;

    std::cout << "H(X:Y): " << std::setprecision(16) << mutualEntropy << std::endl << std::endl;

    return;

}


void LinearOpticalTransform::setToFullHilbertSpace(int subPhotons, int subModes,Eigen::MatrixXi& nv){

    if(subPhotons==0 && subModes == 0){

        nv.resize(0,0);

        return;

    }

    int markers = subPhotons + subModes - 1;
    int myints[markers];
    int i = 0;
    while(i<subPhotons){
        myints[i]=1;
        i++;
    }
    while(i<markers){
        myints[i]=0;
        i++;
    }
    nv = Eigen::MatrixXi::Zero(g(subPhotons,subModes),subModes);
    i = 0;
    int j,k = 0;
    do {
        j = 0;
        k = 0;
        while(k<markers){
        if(myints[k]==1){
            nv(i,j)=nv(i,j)+1;
        }
        else if(myints[k]==0){
            j++;
        }

        k++;
        }
        i++;
    } while ( std::prev_permutation(myints,myints+markers) );
    return;;
}


int LinearOpticalTransform::g(const int& n,const int& m){
    if(n==0 && m==0){
        return 0;
    }
    else if(n==0 && m>0){
        return 1;
    }

    else{
        return (int)(doublefactorial(n+m-1)/(doublefactorial(n)*doublefactorial(m-1))+0.5);
    }
}

double LinearOpticalTransform::doublefactorial(int x){

    assert(x < 171);

    double total=1.0;
    if (x>=0){
        for(int i=x;i>0;i--){
            total=i*total;
        }
    }
    else{
        std::cout << "invalid factorial" << std::endl;
        total=-1;
    }
    return total;
}

//LinearOpticalTransform::LinearOpticalTransform(){
//
//
//}
//
//void LinearOpticalTransform::setOmega(Eigen::MatrixXcd& U){
//
//    for(int i=0;i<omega.rows();i++){
//
//        omega.row(i) = Eigen::VectorXcd::Zero(omega.cols());
//
//        do{
//
//            for(int j=0;j<omega.cols();j++){
//
//                std::complex<double> Uprod(1.0,0.0);
//
//                for(int k=0;k<m[j].size();k++){
//
//
//                    Uprod *= U( m[j][k],mPrime[i][k] );
//
//                }
//
//                omega(i,j) += Uprod;
//
//            }
//
//        } while( std::next_permutation( mPrime[i].begin(), mPrime[i].end() ) );
//
//        double bosonNum = 1.0;
//
//        for(int p=0;p<U.rows();p++) bosonNum *= factorial[ nPrime[i][p] ];
//
//        for(int j=0;j<omega.cols();j++){
//
//            double bosonDen = 1.0;
//
//            for(int p=0;p<U.rows();p++) bosonDen *= factorial[ n[j][p] ];
//
//            omega(i,j) *= sqrt( bosonNum/bosonDen );
//
//        }
//
//    }
//
//    return;
//
//}
//
//void LinearOpticalTransform::initializeCircuit(Eigen::MatrixXi& inBasis, Eigen::MatrixXi& outBasis){
//
//    omega.resize( outBasis.rows(), inBasis.rows() );
//
//    n.resize( inBasis.rows() );
//    m.resize( inBasis.rows() );
//
//    nPrime.resize( outBasis.rows() );
//    mPrime.resize( outBasis.rows() );
//
//    int photons = inBasis.row(0).sum();
//
//    for(int i=0;i<inBasis.rows();i++){
//
//        assert( inBasis.row(i).sum() == photons && "Error: Photon number must be preserved you have included some input basis states that do not have the correct number of photons." );
//
//        n.at(i).resize( inBasis.cols() );
//
//        for(int j=0;j<inBasis.cols();j++) n.at(i).at(j) = inBasis(i,j);
//
//        m.at(i).resize( photons );
//
//        setmVec( m.at(i), n.at(i) );
//
//    }
//
//    inBasis.resize(0,0);
//
//    for(int i=0;i<outBasis.rows();i++){
//
//        assert( outBasis.row(i).sum() == photons && "Error: Photon number must be preserved you have included some output basis states that do not have the correct number of photons." );
//
//        nPrime.at(i).resize( outBasis.cols() );
//
//        for(int j=0;j<outBasis.cols();j++) nPrime.at(i).at(j) = outBasis(i,j);
//
//        mPrime.at(i).resize( photons );
//
//        setmVec( mPrime.at(i), nPrime.at(i) );
//
//    }
//
//    outBasis.resize(0,0);
//
//    factorial.resize( photons + 1 );
//
//    for(int i=0;i<factorial.size();i++) factorial[i] = doublefactorial(i);
//
//    return;
//
//}
//
//void LinearOpticalTransform::setmVec(std::vector<int>& m, std::vector<int>& n){
//
//    int k=0;
//
//    for(int i=0;i<n.size();i++){
//
//        for(int j=0;j<n.at(i);j++){
//
//            m.at(k) = i;
//
//            k++;
//
//        }
//
//    }
//
//    return;
//}
//
//template <typename T>
//void LinearOpticalTransform::printVec(std::vector<T>& a){
//
//    for(int i=0;i<a.size();i++) std::cout << a[i] << "\t";
//
//    std::cout << std::endl;
//
//    return;
//
//}
//
//double LinearOpticalTransform::doublefactorial(int x){
//
//    assert(x < 171);
//
//    double total=1.0;
//    if (x>=0){
//        for(int i=x;i>0;i--){
//            total=i*total;
//        }
//    }
//    else{
//        std::cout << "invalid factorial" << std::endl;
//        total=-1;
//    }
//    return total;
//}
