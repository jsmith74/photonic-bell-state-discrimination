#ifndef UGENERATION_H_INCLUDED
#define UGENERATION_H_INCLUDED

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <iomanip>

class UGeneration{

    public:

        UGeneration();
        void initializeUCondition1(Eigen::MatrixXcd& U);
        void setZeroEntryQuant(Eigen::MatrixXcd& U);

        double zeroEntryQuant;
        Eigen::MatrixXi zeroEntries;

    private:

        void initializeZeroEntries();
        void setZeroEntriesRandomlyCondition1(int ancillaRows);
        bool checkCondition1();



};

#endif // UGENERATION_H_INCLUDED
