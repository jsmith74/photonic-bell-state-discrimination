#include "LinearOpticalTransform.h"

LinearOpticalTransform::LinearOpticalTransform(){



}

void LinearOpticalTransform::checkThreadsAndProcs(){

    int numProcs = omp_get_num_procs();

#pragma omp parallel
{

    int numThreads = omp_get_num_threads();

    assert( numThreads == numProcs );

}

    return;

}


void LinearOpticalTransform::setParallelGrid(){

    int numProcs = omp_get_num_procs();

    int nPrime[ 4 + ancillaModes ];
    int mPrime[ 2 + ancillaPhotons ];

    nPrime[0] = 2 + ancillaPhotons;

    for(int i=1;i<4+ancillaModes;i++) nPrime[i] = 0;

    setMPrime( &nPrime[0],&mPrime[0] );

    int k = 0;

    for(int y=0;y<HSDimension;y++){

        do{

            assert( k < 2147483645 );

            k++;

        } while( std::next_permutation( &mPrime[0],&mPrime[ancillaPhotons+2] ) );

        iterateNPrime( &nPrime[0],&nPrime[4+ancillaModes] );

        setMPrime( &nPrime[0],&mPrime[0] );

    }

    termsPerThread = k / numProcs;

    assert( termsPerThread * numProcs == k );

    // write a parallelization for this - divvy up the work appropriately (as evenly as possible and sequential without having state amplitudes crossing over omp boundaries)
    // only iterate through the ones you have to on a particular thread;
    //

    return;

}

void LinearOpticalTransform::initializeCircuit(int& ancillaP,int& ancillaM){

    ancillaPhotons = ancillaP;
    ancillaModes = ancillaM;
    HSDimension = g( ancillaPhotons + 2,ancillaModes + 4 );

    factorial.resize( ancillaPhotons + 2 + 1 );

    for(int i=0;i<factorial.size();i++) factorial[i] = doublefactorial(i);

    assert( ancillaModes >= ancillaPhotons );

    checkThreadsAndProcs();

    setParallelGrid();

    return;

}


void LinearOpticalTransform::setMutualEntropy(Eigen::MatrixXcd& U){

    double pyx[4];

    double parallelMutualEntropy = 0;

    double totalPyx0 = 0;
    double totalPyx1 = 0;
    double totalPyx2 = 0;
    double totalPyx3 = 0;

    /** ===== NOTE ==============================================

            THE REDUCTION OVER totalPyxN IS UNNESSESSARY FOR THE
            CASE OF UNITARY U - REMOVE THOSE TERMS FROM THE REDUCTION PRAGMA IN THOSE CASES (THE REDUCTION IS KIND OF EXPENSIVE)

        ========================================================= */

//#pragma omp parallel reduction(+:parallelMutualEntropy,totalPyx0,totalPyx1,totalPyx2,totalPyx3)
{

    int numbThreads = omp_get_num_threads();
    int threadID = omp_get_thread_num();

    int k = 0;

    int nPrime[ 4 + ancillaModes ];
    int mPrime[ 2 + ancillaPhotons ];

    nPrime[0] = 2 + ancillaPhotons;

    for(int i=1;i<4+ancillaModes;i++) nPrime[i] = 0;

    setMPrime( &nPrime[0],&mPrime[0] );

    for(int y=0;y<HSDimension;y++){

        std::complex<double> stateAmplitude[4];

        stateAmplitude[0] = 0.0;
        stateAmplitude[1] = 0.0;
        stateAmplitude[2] = 0.0;
        stateAmplitude[3] = 0.0;

        do{

            if( k % numbThreads == threadID ){

                setStateAmplitude(stateAmplitude,U,mPrime);

            }

            k++;

        } while( std::next_permutation( &mPrime[0],&mPrime[ancillaPhotons+2] ) );

        normalizeStateAmplitude(stateAmplitude,nPrime);

        pyx[0] = std::norm( stateAmplitude[0] );
        pyx[1] = std::norm( stateAmplitude[1] );
        pyx[2] = std::norm( stateAmplitude[2] );
        pyx[3] = std::norm( stateAmplitude[3] );

        if(pyx[0] != 0.0) parallelMutualEntropy += pyx[0] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[0] );
        if(pyx[1] != 0.0) parallelMutualEntropy += pyx[1] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[1] );
        if(pyx[2] != 0.0) parallelMutualEntropy += pyx[2] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[2] );
        if(pyx[3] != 0.0) parallelMutualEntropy += pyx[3] * log2( ( pyx[0] + pyx[1] + pyx[2] + pyx[3] ) / pyx[3] );

        totalPyx0 += pyx[0];
        totalPyx1 += pyx[1];
        totalPyx2 += pyx[2];
        totalPyx3 += pyx[3];

        iterateNPrime( &nPrime[0],&nPrime[4+ancillaModes] );

        setMPrime( &nPrime[0],&mPrime[0] );

    }

}

    totalPyx0 = 1 - totalPyx0;
    totalPyx1 = 1 - totalPyx1;
    totalPyx2 = 1 - totalPyx2;
    totalPyx3 = 1 - totalPyx3;

    double logNum = totalPyx0 + totalPyx1 + totalPyx2 + totalPyx3;

    if(totalPyx0 > 0 && logNum > 0) parallelMutualEntropy += totalPyx0 * log2( logNum / totalPyx0 );
    if(totalPyx1 > 0 && logNum > 0) parallelMutualEntropy += totalPyx1 * log2( logNum / totalPyx1 );
    if(totalPyx2 > 0 && logNum > 0) parallelMutualEntropy += totalPyx2 * log2( logNum / totalPyx2 );
    if(totalPyx3 > 0 && logNum > 0) parallelMutualEntropy += totalPyx3 * log2( logNum / totalPyx3 );

    mutualEntropy = parallelMutualEntropy;

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



bool LinearOpticalTransform::iterateNPrime(int* __begin,int* __end){

    int* ptr = __end - 2;

    while( *ptr == 0 ){

        if( ptr == __begin ) return false;

        ptr--;

    }

    *ptr -= 1;

    *( ptr + 1 ) = *( __end -1 ) + 1;

    if( ptr + 1 != __end - 1 ) *( __end - 1 ) = 0;

    return true;

}

void LinearOpticalTransform::setMPrime( int* __nBegin, int* __mBegin ){

    int k=0;

    for(int i=0;i<ancillaModes+4;i++) for(int j=0;j < *(__nBegin + i);j++){

            *( __mBegin + k ) = i;

            k++;

    }

    return;

}

