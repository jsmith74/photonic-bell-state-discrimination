#ifndef UGENERATION_H_INCLUDED
#define UGENERATION_H_INCLUDED

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <iomanip>
#include "BFGS_Optimization.h"

class UGeneration{

    public:

        UGeneration();

        void testStandardUnitary(Eigen::MatrixXcd& U);
        void testRandomUnitary(Eigen::MatrixXcd& U);
        void initializeUCondition1(Eigen::MatrixXcd& U);
        void testRandomUCondition1(Eigen::MatrixXcd& U);
        void setUCondition1(Eigen::VectorXd& position,Eigen::MatrixXcd& U);
        int setFuncDimension();

    private:

        Eigen::MatrixXi zeroEntries;
        Eigen::MatrixXd zeroEntriesDouble;

        void initializeZeroEntries();
        void setZeroEntriesRandomlyCondition1(int ancillaRows);
        bool checkCondition1();

        void setPosition1( Eigen::VectorXd& position );

};

#endif // UGENERATION_H_INCLUDED
