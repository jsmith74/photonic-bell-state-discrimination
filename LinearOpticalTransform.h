#ifndef LINEAROPTICALTRANSFORM_H_INCLUDED
#define LINEAROPTICALTRANSFORM_H_INCLUDED

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <fstream>


class LinearOpticalTransform{

    public:

        double mutualEntropy;
        LinearOpticalTransform();
        void initializeCircuit(int& ancillaP,int& ancillaM);
        void setMutualEntropy(Eigen::MatrixXcd& U);

    private:

        int ancillaPhotons, ancillaModes, HSDimension, termsPerThread, numProcs;
        std::vector<double> factorial;
        std::vector<int> parallelGrid;

        int g(const int& n,const int& m);
        double doublefactorial(int x);
        void setToFullHilbertSpace(int subPhotons, int subModes,Eigen::MatrixXi& nv);
        void setNPrimeAndMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);
        void setmVec(std::vector<int>& m, std::vector<int>& n);
        inline void setStateAmplitude(std::complex<double> stateAmplitude[],Eigen::MatrixXcd& U,int mPrime[]);
        inline void normalizeStateAmplitude(std::complex<double> stateAmplitude[],int nPrime[]);

        void setMPrime( int* __nBegin, int* __mBegin );
        bool iterateNPrime(int* __begin,int* __end);

        void checkThreadsAndProcs();
        void setParallelGrid();

        template<typename T>
        void printVec(std::vector<T>& vec);
};

inline void LinearOpticalTransform::normalizeStateAmplitude(std::complex<double> stateAmplitude[],int nPrime[]){

    stateAmplitude[0] *= 0.7071067811865475;
    stateAmplitude[1] *= 0.7071067811865475;
    stateAmplitude[2] *= 0.7071067811865475;
    stateAmplitude[3] *= 0.7071067811865475;

    for(int p=0;p<ancillaModes+4;p++){

        stateAmplitude[0] *= sqrt( factorial[ nPrime[p] ] );
        stateAmplitude[1] *= sqrt( factorial[ nPrime[p] ] );
        stateAmplitude[2] *= sqrt( factorial[ nPrime[p] ] );
        stateAmplitude[3] *= sqrt( factorial[ nPrime[p] ] );

    }

    return;

}

inline void LinearOpticalTransform::setStateAmplitude(std::complex<double> stateAmplitude[],Eigen::MatrixXcd& U,int mPrime[]){

    std::complex<double> UProdTemp(1.0,0.0);

    for(int i=0;i<ancillaPhotons;i++) UProdTemp *= U( i,mPrime[i] );

    stateAmplitude[0] += UProdTemp * ( U(ancillaModes,mPrime[ancillaPhotons]) * U(ancillaModes+2,mPrime[ancillaPhotons+1])
                                    + U(ancillaModes + 1,mPrime[ancillaPhotons]) * U(ancillaModes + 3,mPrime[ancillaPhotons+1]) );

    stateAmplitude[1] += UProdTemp * ( U(ancillaModes,mPrime[ancillaPhotons]) * U(ancillaModes+3,mPrime[ancillaPhotons+1])
                                    + U(ancillaModes + 1,mPrime[ancillaPhotons]) * U(ancillaModes + 2,mPrime[ancillaPhotons+1]) );

    stateAmplitude[2] += UProdTemp * ( U(ancillaModes,mPrime[ancillaPhotons]) * U(ancillaModes+2,mPrime[ancillaPhotons+1])
                                    - U(ancillaModes + 1,mPrime[ancillaPhotons]) * U(ancillaModes + 3,mPrime[ancillaPhotons+1]) );

    stateAmplitude[3] += UProdTemp * ( U(ancillaModes,mPrime[ancillaPhotons]) * U(ancillaModes+3,mPrime[ancillaPhotons+1])
                                    - U(ancillaModes + 1,mPrime[ancillaPhotons]) * U(ancillaModes + 2,mPrime[ancillaPhotons+1]) );

    return;

}

#endif // LINEAROPTICALTRANSFORM_H_INCLUDED
