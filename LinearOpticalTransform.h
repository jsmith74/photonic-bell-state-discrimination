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
        int* nPrime;
        int* mPrime;

        int g(const int& n,const int& m);
        double doublefactorial(int x);
        void setToFullHilbertSpace(int subPhotons, int subModes,Eigen::MatrixXi& nv);
        void setNPrimeAndMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime);
        void setmVec(std::vector<int>& m, std::vector<int>& n);

        void setMPrime( int* __nBegin, int* __mBegin );
        bool iterateNPrime(int* __begin,int* __end);

        void checkThreadsAndProcs();
        void setParallelGrid();

        template<typename T>
        void printVec(std::vector<T>& vec);
};


#endif // LINEAROPTICALTRANSFORM_H_INCLUDED
