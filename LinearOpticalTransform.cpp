#include "LinearOpticalTransform.h"



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

    OffloadtoGPU.queryGPUDevices();

    OffloadtoGPU.setGPUDevice( 0 );

    OffloadtoGPU.numberOfTerms = evaluateNumberOfTerms();

    OffloadtoGPU.allocateResources();

    OffloadtoGPU.sendFactorialToGPU( factorial );

    OffloadtoGPU.initializeStartingNPrimeMPrime(nPrime,mPrime);

    nPrime.resize( 0 );
    mPrime.resize( 0 );

    return;

}

int LinearOpticalTransform::evaluateNumberOfTerms(){

    int k = 0;

    for(int y=0;y<HSDimension;y++) do{

        assert( k < 2147483645 );

        k++;

    } while( std::next_permutation( mPrime[y].begin() , mPrime[y].end() ) );

    return k;

}

void LinearOpticalTransform::setMutualEntropy(Eigen::MatrixXcd& U){

    OffloadtoGPU.sendUToGPU( U );

    mutualEntropy = OffloadtoGPU.setMutualEntropy();

//    double pyx[4];
//
//    mutualEntropy = 0.0;
//
//    double totalPyx[4];
//    totalPyx[0] = 0;
//    totalPyx[1] = 0;
//    totalPyx[2] = 0;
//    totalPyx[3] = 0;
//
//    for(int y=0;y<HSDimension;y++){
//
//        std::complex<double> stateAmplitude[4];
//
//        stateAmplitude[0] = 0.0;
//        stateAmplitude[1] = 0.0;
//        stateAmplitude[2] = 0.0;
//        stateAmplitude[3] = 0.0;
//
//        do{
//
//            setStateAmplitude(stateAmplitude,U,y);
//
//        } while( std::next_permutation( mPrime[y].begin(),mPrime[y].end() ) );
//
//        normalizeStateAmplitude(stateAmplitude,y);
//
//        pyx[0] = std::norm( stateAmplitude[0] );
//        pyx[1] = std::norm( stateAmplitude[1] );
//        pyx[2] = std::norm( stateAmplitude[2] );
//        pyx[3] = std::norm( stateAmplitude[3] );
//
//        if(pyx[0] != 0.0) mutualEntropy += pyx[0] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[0] );
//        if(pyx[1] != 0.0) mutualEntropy += pyx[1] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[1] );
//        if(pyx[2] != 0.0) mutualEntropy += pyx[2] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[2] );
//        if(pyx[3] != 0.0) mutualEntropy += pyx[3] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[3] );
//
//        totalPyx[0] += pyx[0];
//        totalPyx[1] += pyx[1];
//        totalPyx[2] += pyx[2];
//        totalPyx[3] += pyx[3];
//
//    }
//
//    totalPyx[0] = 1 - totalPyx[0];
//    totalPyx[1] = 1 - totalPyx[1];
//    totalPyx[2] = 1 - totalPyx[2];
//    totalPyx[3] = 1 - totalPyx[3];
//
//    double logNum = totalPyx[0] + totalPyx[1] + totalPyx[2] + totalPyx[3];
//
//    if(totalPyx[0] > 0 && logNum > 0) mutualEntropy += totalPyx[0] * log2( logNum / totalPyx[0] );
//    if(totalPyx[1] > 0 && logNum > 0) mutualEntropy += totalPyx[1] * log2( logNum / totalPyx[1] );
//    if(totalPyx[2] > 0 && logNum > 0) mutualEntropy += totalPyx[2] * log2( logNum / totalPyx[2] );
//    if(totalPyx[3] > 0 && logNum > 0) mutualEntropy += totalPyx[3] * log2( logNum / totalPyx[3] );

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


//#define ANCILLA_PHOTONS 6
//#define ANCILLA_MODES 8

//bool iterateNPrime(int* __begin,int* __end){
//
//    int* ptr = __end - 2;
//
//    while( *ptr == 0 ){
//
//        if( ptr == __begin ) return false;
//
//        ptr--;
//
//    }
//
//    *ptr -= 1;
//
//    *( ptr + 1 ) = *( __end -1 ) + 1;
//
//    if( ptr + 1 != __end - 1 ) *( __end - 1 ) = 0;
//
//    return true;
//
//}
//
//void setMPrime( int* __nBegin, int* __mBegin ){
//
//    int k=0;
//
//    for(int i=0;i<ANCILLA_MODES+4;i++) for(int j=0;j < *(__nBegin + i);j++){
//
//            *( __mBegin + k ) = i;
//
//            k++;
//
//    }
//
//    return;
//
//}

LinearOpticalTransform::LinearOpticalTransform(){

    /** === TESTING MY ITERATE N FUNCTION ======= */

//    int photons = 8;
//    int modes = 12;
//
//    Eigen::MatrixXi outBasis;
//
//    setToFullHilbertSpace( photons, modes, outBasis);
//
//    std::cout << outBasis << std::endl << std::endl;
//
//    int nPrimeTest[ modes ];
//    int mPrimeTest[ photons ];
//
//    for(int i=0;i<modes;i++) nPrimeTest[i] = 0;
//    nPrimeTest[0] = photons;
//
//    int k = 0;
//
//    do{
//
//        for(int i=0;i<modes;i++) std::cout << nPrimeTest[i] << " ";
//        std::cout << std::endl;
//
//        setMPrime( &nPrimeTest[0],&mPrimeTest[0] );
//
//        for(int i=0;i<photons;i++) std::cout << mPrimeTest[i] << " ";
//        std::cout << std::endl;
//
//        std::vector<int> mPrimeKnownCompare,nPrimeKnownCompare;
//        mPrimeKnownCompare.resize( photons );
//        nPrimeKnownCompare.resize( modes );
//        for(int i=0;i<modes;i++) nPrimeKnownCompare[i] = outBasis(k,i);
//        setmVec(mPrimeKnownCompare,nPrimeKnownCompare);
//
//
//        for(int i=0;i<photons;i++) std::cout << mPrimeKnownCompare[i] << " ";
//        std::cout << std::endl;
//
//        for(int i=0;i<modes;i++) assert( outBasis(k,i) == nPrimeTest[i] );
//        for(int i=0;i<photons;i++) assert( mPrimeKnownCompare[i] == mPrimeTest[i] );
//
//        k++;
//
//    } while( iterateNPrime(&nPrimeTest[0],&nPrimeTest[modes]) );
//
//    assert( k = outBasis.rows() );
//
//    assert( false && "SUCCESS");


    /** ========================================= */

}
