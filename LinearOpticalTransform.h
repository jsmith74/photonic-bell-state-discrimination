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
        void initializeCircuit(Eigen::MatrixXi& inBasis, Eigen::MatrixXi& outBasis);
        void setOmega(Eigen::MatrixXcd& U);

        Eigen::MatrixXcd omega;

    private:

        std::vector< std::vector<int> > n,m,nPrime,mPrime;
        std::vector<double> factorial;

        void setmVec(std::vector<int>& m, std::vector<int>& n);

        template <typename T>
        void printVec(std::vector<T>& a);

        double doublefactorial(int x);


};

#endif // LINEAROPTICALTRANSFORM_H_INCLUDED
