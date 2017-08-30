#ifndef LINEAROPTICALTRANSFORM_H_INCLUDED
#define LINEAROPTICALTRANSFORM_H_INCLUDED

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <iomanip>

class LinearOpticalTransform{

    public:

        LinearOpticalTransform();
        void initializeCircuit(int& ancillaP,int& ancillaM);
        void setMutualEntropy(Eigen::MatrixXcd& U);

    private:

        int ancillaPhotons, ancillaModes, HSDimension;
        std::vector<double> factorial;
        std::vector< std::vector<int> > nPrime, mPrime;

        double mutualEntropy;

        int g(const int& n,const int& m);
        double doublefactorial(int x);
        void setToFullHilbertSpace(int subPhotons, int subModes,Eigen::MatrixXi& nv);
        void setNPrimeAndMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);
        void setmVec(std::vector<int>& m, std::vector<int>& n);
        inline void setStateAmplitude(std::complex<double> stateAmplitude[],Eigen::MatrixXcd& U,int& y);
        inline void normalizeStateAmplitude(std::complex<double> stateAmplitude[],int& y);

        template<typename T>
        void printVec(std::vector<T>& vec);
};

inline void LinearOpticalTransform::normalizeStateAmplitude(std::complex<double> stateAmplitude[],int& y){

    stateAmplitude[0] *= 1.0 / sqrt(2.0);
    stateAmplitude[1] *= 1.0 / sqrt(2.0);
    stateAmplitude[2] *= 1.0 / sqrt(2.0);
    stateAmplitude[3] *= 1.0 / sqrt(2.0);

    for(int p=0;p<nPrime[y].size();p++){

        stateAmplitude[0] *= sqrt( factorial[ nPrime[y][p] ] );
        stateAmplitude[1] *= sqrt( factorial[ nPrime[y][p] ] );
        stateAmplitude[2] *= sqrt( factorial[ nPrime[y][p] ] );
        stateAmplitude[3] *= sqrt( factorial[ nPrime[y][p] ] );

    }

    return;

}

inline void LinearOpticalTransform::setStateAmplitude(std::complex<double> stateAmplitude[],Eigen::MatrixXcd& U,int& y){

    std::complex<double> UProdTemp(1.0,0.0);

    for(int i=0;i<ancillaPhotons;i++) UProdTemp *= U( i,mPrime[y][i] );

    stateAmplitude[0] += UProdTemp * ( U(ancillaModes,mPrime[y][ancillaPhotons]) * U(ancillaModes+2,mPrime[y][ancillaPhotons+1])
                                    + U(ancillaModes + 1,mPrime[y][ancillaPhotons]) * U(ancillaModes + 3,mPrime[y][ancillaPhotons+1]) );

    stateAmplitude[1] += UProdTemp * ( U(ancillaModes,mPrime[y][ancillaPhotons]) * U(ancillaModes+3,mPrime[y][ancillaPhotons+1])
                                    + U(ancillaModes + 1,mPrime[y][ancillaPhotons]) * U(ancillaModes + 2,mPrime[y][ancillaPhotons+1]) );

    stateAmplitude[2] += UProdTemp * ( U(ancillaModes,mPrime[y][ancillaPhotons]) * U(ancillaModes+2,mPrime[y][ancillaPhotons+1])
                                    - U(ancillaModes + 1,mPrime[y][ancillaPhotons]) * U(ancillaModes + 3,mPrime[y][ancillaPhotons+1]) );

    stateAmplitude[3] += UProdTemp * ( U(ancillaModes,mPrime[y][ancillaPhotons]) * U(ancillaModes+3,mPrime[y][ancillaPhotons+1])
                                    - U(ancillaModes + 1,mPrime[y][ancillaPhotons]) * U(ancillaModes + 2,mPrime[y][ancillaPhotons+1]) );
    return;

}

#endif // LINEAROPTICALTRANSFORM_H_INCLUDED
