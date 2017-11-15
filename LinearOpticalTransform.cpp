#include "LinearOpticalTransform.h"

#define ALLOC alloc_if(1)
#define FREE free_if(1)
#define RETAIN free_if(0)
#define REUSE alloc_if(0)

/** == TOGGLE DELETING MEMORY ON CPU =========== */

//#define DELETE_CPU_ARRAYS
#define ANCILLA_PHOTONS 6
#define ANCILLA_MODES 6

/** ============================================ */

__declspec(target(mic)) bool dev_next_permutation(int* __first, int* __last);

__declspec(target(mic)) int* dev_parallelGrid;
__declspec(target(mic)) int* dev_nPrime;
__declspec(target(mic)) int* dev_mPrime;
__declspec(target(mic)) int* dev_factorial;
__declspec(target(mic)) int dev_numThreadsCPU;


void LinearOpticalTransform::initializeCircuit(int& ancillaP,int& ancillaM,int intParam){

    CPUWorkload = 51100000;

    ancillaPhotons = ancillaP;
    ancillaModes = ancillaM;
    HSDimension = g( ancillaPhotons + 2,ancillaModes + 4 );

    factorial.resize( ancillaPhotons + 2 + 1 );

    for(int i=0;i<factorial.size();i++) factorial[i] = doublefactorial(i);

    dev_factorial = new int[ ancillaPhotons + 2 + 1 ];

    for(int i=0;i<factorial.size();i++) dev_factorial[i] = factorial[i];

    factorial.resize(0);

    assert( ancillaModes >= ancillaPhotons );

    checkThreadsAndProcs();

    setParallelGrid();

    dev_nPrime = new int[ HSDimension * (4 + ancillaModes) ];
    dev_mPrime = new int[ HSDimension * (2 + ancillaPhotons) ];

    std::vector< std::vector<int> > nPrimeTemp,mPrimeTemp;

    nPrimeTemp.resize( HSDimension );
    mPrimeTemp.resize( HSDimension );

    for(int i=0;i<HSDimension;i++){ nPrimeTemp.at(i).resize(ancillaModes + 4); mPrimeTemp.at(i).resize(ancillaPhotons + 2); }

    setNPrimeAndMPrime(nPrimeTemp,mPrimeTemp);

    for(int y=0;y<HSDimension;y++){

        for(int i=0;i<4+ancillaModes;i++) dev_nPrime[ y*(4+ancillaModes) + i ] = nPrimeTemp[y][i];

        for(int i=0;i<2+ancillaPhotons;i++) dev_mPrime[ y*(2+ancillaPhotons) + i ] = mPrimeTemp[y][i];

    }

    nPrimeTemp.resize(0);
    mPrimeTemp.resize(0);

#pragma offload target(mic:0) in(dev_nPrime[0:HSDimension * (4 + ancillaModes)] : ALLOC RETAIN ) \
                            in(dev_mPrime[0:HSDimension * (2 + ancillaPhotons)] : ALLOC RETAIN ) \
                            in(dev_factorial[0:ancillaPhotons + 2 + 1] : ALLOC RETAIN )
#pragma omp parallel
{

}

#pragma offload target(mic:1) in(dev_nPrime[0:HSDimension * (4 + ancillaModes)] : ALLOC RETAIN ) \
                            in(dev_mPrime[0:HSDimension * (2 + ancillaPhotons)] : ALLOC RETAIN ) \
                            in(dev_factorial[0:ancillaPhotons + 2 + 1] : ALLOC RETAIN )
#pragma omp parallel
{

}


    #ifdef DELETE_CPU_ARRAYS

        delete[] dev_nPrime;
        delete[] dev_mPrime;
        delete[] dev_factorial;

    #endif

    return;

}

__declspec(target(mic)) inline double Uel(double* U,int& row,int& col,bool imagPart){

    return *( U + 2 * row + 2 * col * (4+ANCILLA_MODES) + imagPart );

}

__declspec(target(mic)) inline double* UelPtr(double* U,int row,int col){

    return ( U + 2 * row + 2 * col * (4+ANCILLA_MODES) );

}

__declspec(target(mic)) inline void complex_prod(double* result,double* c1,double* c2){

    result[0] = c1[0] * c2[0] - c1[1] * c2[1];
    result[1] = c1[1] * c2[0] + c1[0] * c2[1];

    return;

}

__declspec(target(mic)) inline void complex_sum(double* result,double* c1,double* c2){

    result[0] = c1[0] + c2[0];
    result[1] = c1[1] + c2[1];

    return;

}

__declspec(target(mic)) inline void complex_sum_compound(double* result,double* c1){

    result[0] += c1[0];
    result[1] += c1[1];

    return;

}

__declspec(target(mic)) inline void complex_prod_compound(double* result,double* c1){

    double temp = result[0];

    result[0] = result[0] * c1[0] - result[1] * c1[1];
    result[1] = result[1] * c1[0] + temp * c1[1];

    return;

}

__declspec(target(mic)) inline void complex_special_op_plus(double* result,double* c1,double* c2,double* c3,double* c4,double* cProd){

    double temp[2];

    temp[0] = c1[0] * c2[0] - c1[1] * c2[1] + c3[0] * c4[0] - c3[1] * c4[1];

    temp[1] = c1[0] * c2[1] + c1[1] * c2[0] + c3[1] * c4[0] + c3[0] * c4[1];

    result[0] += cProd[0] * temp[0] - cProd[1] * temp[1];

    result[1] += cProd[1] * temp[0] + cProd[0] * temp[1];

    return;

}

__declspec(target(mic)) inline void complex_special_op_minus(double* result,double* c1,double* c2,double* c3,double* c4,double* cProd){

    double temp[2];

    temp[0] = c1[0] * c2[0] - c1[1] * c2[1] - c3[0] * c4[0] + c3[1] * c4[1];

    temp[1] = c1[0] * c2[1] + c1[1] * c2[0] - c3[1] * c4[0] - c3[0] * c4[1];

    result[0] += cProd[0] * temp[0] - cProd[1] * temp[1];

    result[1] += cProd[1] * temp[0] + cProd[0] * temp[1];

    return;

}

void LinearOpticalTransform::setMutualEntropy(Eigen::MatrixXcd& U){

    double parallelMutualEntropyMic0;

    double totalPyx0mic0;
    double totalPyx1mic0;
    double totalPyx2mic0;
    double totalPyx3mic0;

    double* dev_U = (double*)U.data();

    char signal_var0, signal_var1;

#pragma offload signal(&signal_var0) target (mic:0) \
                    out(parallelMutualEntropyMic0,totalPyx0mic0,totalPyx1mic0,totalPyx2mic0,totalPyx3mic0) \
                    in(dev_U[0:2*(4+ANCILLA_MODES)*(4+ANCILLA_MODES)] :  ALLOC FREE ) \
                    nocopy(dev_parallelGrid[0:pGridSize] : REUSE RETAIN ) \
                    nocopy(dev_nPrime[0:HSDimension * (4 + ANCILLA_MODES)] : REUSE RETAIN ) \
                    nocopy(dev_mPrime[0:HSDimension * (2 + ANCILLA_PHOTONS)] : REUSE RETAIN ) \
                    nocopy(dev_factorial[0:ANCILLA_PHOTONS + 2 + 1] : REUSE RETAIN ) \
                    nocopy(dev_numThreadsCPU : REUSE RETAIN)
#pragma omp parallel reduction(+:parallelMutualEntropyMic0,totalPyx0mic0,totalPyx1mic0,totalPyx2mic0,totalPyx3mic0)
{

    int threadID = omp_get_thread_num() + dev_numThreadsCPU;

    parallelMutualEntropyMic0 = 0;

    totalPyx0mic0 = 0;
    totalPyx1mic0 = 0;
    totalPyx2mic0 = 0;
    totalPyx3mic0 = 0;

    for( int y=dev_parallelGrid[threadID]; y<dev_parallelGrid[threadID+1]; y++ ){

        double stateAmplitude[8];

        stateAmplitude[0] = 0.0;
        stateAmplitude[1] = 0.0;
        stateAmplitude[2] = 0.0;
        stateAmplitude[3] = 0.0;
        stateAmplitude[4] = 0.0;
        stateAmplitude[5] = 0.0;
        stateAmplitude[6] = 0.0;
        stateAmplitude[7] = 0.0;

        do{

            double UProdTemp[2];
            UProdTemp[0] = 1.0;
            UProdTemp[1] = 0.0;

            for(int i=0;i<ANCILLA_PHOTONS;i++) complex_prod_compound( UProdTemp, UelPtr(dev_U,i,dev_mPrime[y * (2+ANCILLA_PHOTONS) + i]) );

            complex_special_op_plus(&stateAmplitude[0],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                                UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_plus(&stateAmplitude[2],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_minus(&stateAmplitude[4],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_minus(&stateAmplitude[6],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

        } while( dev_next_permutation( &dev_mPrime[ y * (2+ANCILLA_PHOTONS) ], &dev_mPrime[ (y+1) * (2+ANCILLA_PHOTONS) ] ) );

        double bosonStat = 1;

        for(int p=0;p<ANCILLA_MODES+4;p++) bosonStat *= dev_factorial[ dev_nPrime[y * (4 + ANCILLA_MODES) + p] ];

        stateAmplitude[0] = bosonStat * 0.5 * (stateAmplitude[0] * stateAmplitude[0] + stateAmplitude[1] * stateAmplitude[1]);
        stateAmplitude[1] = bosonStat * 0.5 * (stateAmplitude[2] * stateAmplitude[2] + stateAmplitude[3] * stateAmplitude[3]);
        stateAmplitude[2] = bosonStat * 0.5 * (stateAmplitude[4] * stateAmplitude[4] + stateAmplitude[5] * stateAmplitude[5]);
        stateAmplitude[3] = bosonStat * 0.5 * (stateAmplitude[6] * stateAmplitude[6] + stateAmplitude[7] * stateAmplitude[7]);

        if(stateAmplitude[0] != 0.0) parallelMutualEntropyMic0 += stateAmplitude[0] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[0] );
        if(stateAmplitude[1] != 0.0) parallelMutualEntropyMic0 += stateAmplitude[1] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[1] );
        if(stateAmplitude[2] != 0.0) parallelMutualEntropyMic0 += stateAmplitude[2] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[2] );
        if(stateAmplitude[3] != 0.0) parallelMutualEntropyMic0 += stateAmplitude[3] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[3] );

        totalPyx0mic0 += stateAmplitude[0];
        totalPyx1mic0 += stateAmplitude[1];
        totalPyx2mic0 += stateAmplitude[2];
        totalPyx3mic0 += stateAmplitude[3];

    }

}

    double parallelMutualEntropyMic1;

    double totalPyx0mic1;
    double totalPyx1mic1;
    double totalPyx2mic1;
    double totalPyx3mic1;

#pragma offload signal(&signal_var1) target (mic:1) \
                    out(parallelMutualEntropyMic1,totalPyx0mic1,totalPyx1mic1,totalPyx2mic1,totalPyx3mic1) \
                    in(dev_U[0:2*(4+ANCILLA_MODES)*(4+ANCILLA_MODES)] :  ALLOC FREE ) \
                    nocopy(dev_parallelGrid[0:pGridSize] : REUSE RETAIN ) \
                    nocopy(dev_nPrime[0:HSDimension * (4 + ANCILLA_MODES)] : REUSE RETAIN ) \
                    nocopy(dev_mPrime[0:HSDimension * (2 + ANCILLA_PHOTONS)] : REUSE RETAIN ) \
                    nocopy(dev_factorial[0:ANCILLA_PHOTONS + 2 + 1] : REUSE RETAIN ) \
                    nocopy(dev_numThreadsCPU : REUSE RETAIN)
#pragma omp parallel reduction(+:parallelMutualEntropyMic1,totalPyx0mic1,totalPyx1mic1,totalPyx2mic1,totalPyx3mic1)
{

    int threadID = omp_get_thread_num() + omp_get_num_threads() + dev_numThreadsCPU;

    parallelMutualEntropyMic1 = 0;

    totalPyx0mic1 = 0;
    totalPyx1mic1 = 0;
    totalPyx2mic1 = 0;
    totalPyx3mic1 = 0;

    for( int y=dev_parallelGrid[threadID]; y<dev_parallelGrid[threadID+1]; y++ ){

        double stateAmplitude[8];

        stateAmplitude[0] = 0.0;
        stateAmplitude[1] = 0.0;
        stateAmplitude[2] = 0.0;
        stateAmplitude[3] = 0.0;
        stateAmplitude[4] = 0.0;
        stateAmplitude[5] = 0.0;
        stateAmplitude[6] = 0.0;
        stateAmplitude[7] = 0.0;

        do{

            double UProdTemp[2];
            UProdTemp[0] = 1.0;
            UProdTemp[1] = 0.0;

            for(int i=0;i<ANCILLA_PHOTONS;i++) complex_prod_compound( UProdTemp, UelPtr(dev_U,i,dev_mPrime[y * (2+ANCILLA_PHOTONS) + i]) );

            complex_special_op_plus(&stateAmplitude[0],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                                UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_plus(&stateAmplitude[2],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_minus(&stateAmplitude[4],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_minus(&stateAmplitude[6],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

        } while( dev_next_permutation( &dev_mPrime[ y * (2+ANCILLA_PHOTONS) ], &dev_mPrime[ (y+1) * (2+ANCILLA_PHOTONS) ] ) );

        double bosonStat = 1;

        for(int p=0;p<ANCILLA_MODES+4;p++) bosonStat *= dev_factorial[ dev_nPrime[y * (4 + ANCILLA_MODES) + p] ];

        stateAmplitude[0] = bosonStat * 0.5 * (stateAmplitude[0] * stateAmplitude[0] + stateAmplitude[1] * stateAmplitude[1]);
        stateAmplitude[1] = bosonStat * 0.5 * (stateAmplitude[2] * stateAmplitude[2] + stateAmplitude[3] * stateAmplitude[3]);
        stateAmplitude[2] = bosonStat * 0.5 * (stateAmplitude[4] * stateAmplitude[4] + stateAmplitude[5] * stateAmplitude[5]);
        stateAmplitude[3] = bosonStat * 0.5 * (stateAmplitude[6] * stateAmplitude[6] + stateAmplitude[7] * stateAmplitude[7]);

        if(stateAmplitude[0] != 0.0) parallelMutualEntropyMic1 += stateAmplitude[0] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[0] );
        if(stateAmplitude[1] != 0.0) parallelMutualEntropyMic1 += stateAmplitude[1] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[1] );
        if(stateAmplitude[2] != 0.0) parallelMutualEntropyMic1 += stateAmplitude[2] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[2] );
        if(stateAmplitude[3] != 0.0) parallelMutualEntropyMic1 += stateAmplitude[3] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[3] );

        totalPyx0mic1 += stateAmplitude[0];
        totalPyx1mic1 += stateAmplitude[1];
        totalPyx2mic1 += stateAmplitude[2];
        totalPyx3mic1 += stateAmplitude[3];

    }

}

    double parallelMutualEntropy = 0;

    double totalPyx0 = 0;
    double totalPyx1 = 0;
    double totalPyx2 = 0;
    double totalPyx3 = 0;


#pragma omp parallel default(none) shared(dev_nPrime,dev_mPrime,dev_U,dev_parallelGrid,dev_factorial) \
        reduction(+:parallelMutualEntropy,totalPyx0,totalPyx1,totalPyx2,totalPyx3)
{

    int threadID = omp_get_thread_num();

    parallelMutualEntropy = 0;

    totalPyx0 = 0;
    totalPyx1 = 0;
    totalPyx2 = 0;
    totalPyx3 = 0;

    for( int y=dev_parallelGrid[threadID]; y<dev_parallelGrid[threadID+1]; y++ ){

        double stateAmplitude[8];

        stateAmplitude[0] = 0.0;
        stateAmplitude[1] = 0.0;
        stateAmplitude[2] = 0.0;
        stateAmplitude[3] = 0.0;
        stateAmplitude[4] = 0.0;
        stateAmplitude[5] = 0.0;
        stateAmplitude[6] = 0.0;
        stateAmplitude[7] = 0.0;

        do{

            double UProdTemp[2];
            UProdTemp[0] = 1.0;
            UProdTemp[1] = 0.0;

            for(int i=0;i<ANCILLA_PHOTONS;i++) complex_prod_compound( UProdTemp, UelPtr(dev_U,i,dev_mPrime[y * (2+ANCILLA_PHOTONS) + i]) );

            complex_special_op_plus(&stateAmplitude[0],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                                UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_plus(&stateAmplitude[2],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_minus(&stateAmplitude[4],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

            complex_special_op_minus(&stateAmplitude[6],UelPtr(dev_U,ANCILLA_MODES,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES+3,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),
                               UelPtr(dev_U,ANCILLA_MODES + 1,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS]),UelPtr(dev_U,ANCILLA_MODES + 2,dev_mPrime[y * (2+ANCILLA_PHOTONS) + ANCILLA_PHOTONS+1]),UProdTemp);

        } while( dev_next_permutation( &dev_mPrime[ y * (2+ANCILLA_PHOTONS) ], &dev_mPrime[ (y+1) * (2+ANCILLA_PHOTONS) ] ) );

        double bosonStat = 1;

        for(int p=0;p<ANCILLA_MODES+4;p++) bosonStat *= dev_factorial[ dev_nPrime[y * (4 + ANCILLA_MODES) + p] ];

        stateAmplitude[0] = bosonStat * 0.5 * (stateAmplitude[0] * stateAmplitude[0] + stateAmplitude[1] * stateAmplitude[1]);
        stateAmplitude[1] = bosonStat * 0.5 * (stateAmplitude[2] * stateAmplitude[2] + stateAmplitude[3] * stateAmplitude[3]);
        stateAmplitude[2] = bosonStat * 0.5 * (stateAmplitude[4] * stateAmplitude[4] + stateAmplitude[5] * stateAmplitude[5]);
        stateAmplitude[3] = bosonStat * 0.5 * (stateAmplitude[6] * stateAmplitude[6] + stateAmplitude[7] * stateAmplitude[7]);

        if(stateAmplitude[0] != 0.0) parallelMutualEntropy += stateAmplitude[0] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[0] );
        if(stateAmplitude[1] != 0.0) parallelMutualEntropy += stateAmplitude[1] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[1] );
        if(stateAmplitude[2] != 0.0) parallelMutualEntropy += stateAmplitude[2] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[2] );
        if(stateAmplitude[3] != 0.0) parallelMutualEntropy += stateAmplitude[3] * log2( ( stateAmplitude[0] + stateAmplitude[1] + stateAmplitude[2] + stateAmplitude[3] ) / stateAmplitude[3] );

        totalPyx0 += stateAmplitude[0];
        totalPyx1 += stateAmplitude[1];
        totalPyx2 += stateAmplitude[2];
        totalPyx3 += stateAmplitude[3];

    }

}

#pragma offload wait(&signal_var0) target (mic:0)
{

}

#pragma offload wait(&signal_var1) target (mic:1)
{

}

    parallelMutualEntropy += parallelMutualEntropyMic0 + parallelMutualEntropyMic1;

    totalPyx0 = 1 - totalPyx0 - totalPyx0mic0 - totalPyx0mic1;
    totalPyx1 = 1 - totalPyx1 - totalPyx1mic0 - totalPyx1mic1;
    totalPyx2 = 1 - totalPyx2 - totalPyx2mic0 - totalPyx2mic1;
    totalPyx3 = 1 - totalPyx3 - totalPyx3mic0 - totalPyx3mic1;

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

void LinearOpticalTransform::checkThreadsAndProcs(){

#pragma omp parallel
{

    numThreadsCPU = omp_get_num_threads();

}

    num_coprocessors = _Offload_number_of_devices();

#pragma offload target (mic:0) inout(numThreadsPhi)
#pragma omp parallel
{

    int threadID = omp_get_thread_num();

    if(threadID == 0) numThreadsPhi = omp_get_num_threads();

}

    numThreadsPhi *= num_coprocessors;

    assert( num_coprocessors == 2 );

    dev_numThreadsCPU = numThreadsCPU;

#pragma offload target(mic:0) in( dev_numThreadsCPU : ALLOC RETAIN )
#pragma omp parallel
{

}

#pragma offload target(mic:1) in( dev_numThreadsCPU : ALLOC RETAIN )
#pragma omp parallel
{

}

    return;

}



void LinearOpticalTransform::setParallelGrid(){

    int nPrime[ 4 + ancillaModes ];
    int mPrime[ 2 + ancillaPhotons ];

    nPrime[0] = 2 + ancillaPhotons;

    for(int i=1;i<4+ancillaModes;i++) nPrime[i] = 0;

    setMPrime( &nPrime[0],&mPrime[0] );

    int k = 0;

    int maxTerms = 0;

    for(int y=0;y<HSDimension;y++){

        int j = 0;

        do{

            assert( k < 2147483645 );

            j++;

            k++;

        } while( std::next_permutation( &mPrime[0],&mPrime[ancillaPhotons+2] ) );

        maxTerms = std::max( j,maxTerms );

        iterateNPrime( &nPrime[0],&nPrime[4+ancillaModes] );

        setMPrime( &nPrime[0],&mPrime[0] );

    }

    assert( CPUWorkload <= k );

    int phiWorkload = k - CPUWorkload;

    int termsPerThreadCPU = ( CPUWorkload + numThreadsCPU - 1 ) / numThreadsCPU;
    int termsPerThreadPhi = ( phiWorkload + numThreadsPhi - 1 ) / numThreadsPhi;

    assert( numThreadsCPU * termsPerThreadCPU + numThreadsPhi * termsPerThreadPhi >= k );
    assert( termsPerThreadCPU >= maxTerms && termsPerThreadPhi >= maxTerms );

    std::cout << "Terms per thread: " << std::endl;
    std::cout << termsPerThreadCPU << "\t" << termsPerThreadPhi << std::endl;

    nPrime[0] = 2 + ancillaPhotons;

    for(int i=1;i<4+ancillaModes;i++) nPrime[i] = 0;

    setMPrime( &nPrime[0],&mPrime[0] );

    int termTracker = 0;

    Eigen::VectorXi tempDist(1);

    Eigen::VectorXi termCounter;

    termCounter.resize(0);

    tempDist(0) = 0;

    bool CPUAlloc = true;

    for(int y=0;y<HSDimension;y++){

        int j = 0;

        do{

            j++;

        } while( std::next_permutation( &mPrime[0],&mPrime[ancillaPhotons+2] ) );

        termTracker += j;

        if( (termTracker >= termsPerThreadCPU || y == HSDimension - 1) && CPUAlloc ){

            tempDist.conservativeResize( tempDist.size() + 1 );
            tempDist( tempDist.size() -1 ) = y + 1;

            termCounter.conservativeResize( termCounter.size() + 1 );
            termCounter( termCounter.size() - 1 ) = termTracker;

            termTracker = 0;

        }

        if( (termTracker >= termsPerThreadPhi || y == HSDimension - 1) && !CPUAlloc ){

            tempDist.conservativeResize( tempDist.size() + 1 );
            tempDist( tempDist.size() -1 ) = y + 1;

            termCounter.conservativeResize( termCounter.size() + 1 );
            termCounter( termCounter.size() - 1 ) = termTracker;

            termTracker = 0;

        }

        if( termCounter.size() == numThreadsCPU ){

            CPUAlloc = false;

        }

        iterateNPrime( &nPrime[0],&nPrime[4+ancillaModes] );

        setMPrime( &nPrime[0],&mPrime[0] );

    }

    std::ofstream outfile("distributionMonitor.dat");

    for(int i=0;i<termCounter.size();i++) outfile << i << "\t" << termCounter(i) << std::endl;

    outfile.close();

    pGridSize = numThreadsCPU + numThreadsPhi + 1;

    dev_parallelGrid = new int[pGridSize];

    assert( tempDist.size() <= pGridSize );

    for(int i=0;i<tempDist.size();i++) dev_parallelGrid[i] = tempDist(i);
    for(int i=tempDist.size();i<pGridSize;i++) dev_parallelGrid[i] = 0;

#pragma offload target(mic:0) in(dev_parallelGrid[0:pGridSize] : ALLOC RETAIN )
#pragma omp parallel
{

}

#pragma offload target(mic:1) in(dev_parallelGrid[0:pGridSize] : ALLOC RETAIN )
#pragma omp parallel
{

}

    #ifdef DELETE_CPU_ARRAYS

        delete[] dev_parallelGrid;

    #endif

    assert( termCounter.sum() == k );

    return;

}

__declspec(target(mic)) inline void iter_swap(int* __a, int* __b) {
  int __tmp = *__a;
  *__a = *__b;
  *__b = __tmp;
}


__declspec(target(mic)) void reverse(int* __first, int* __last) {

  while (true)
    if (__first == __last || __first == --__last)
      return;
    else{
      iter_swap(__first++, __last);
    }
}

__declspec(target(mic)) bool dev_next_permutation(int* __first, int* __last) {

  if (__first == __last)
    return false;
  int* __i = __first;
  ++__i;
  if (__i == __last)
    return false;
  __i = __last;
  --__i;

  for(;;) {
    int* __ii = __i;
    --__i;
    if (*__i < *__ii) {
      int* __j = __last;
      while (!(*__i < *--__j))
        {}
    iter_swap(__i, __j);
      reverse(__ii, __last);
      return true;
    }
    if (__i == __first) {
      reverse(__first, __last);
      return false;
    }
  }

}

LinearOpticalTransform::LinearOpticalTransform(){



}
