#ifndef MERITFUNCTION_H_INCLUDED
#define MERITFUNCTION_H_INCLUDED

#include "LinearOpticalTransform.h"
#include "UGeneration.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unsupported/Eigen/MatrixFunctions>
#include <omp.h>

class MeritFunction{

    public:

        MeritFunction();
        void setMeritFunction(int intParam);
        double f(Eigen::VectorXd& position);
        int funcDimension;
        void printReport(Eigen::VectorXd& position);
        Eigen::VectorXd setInitialPosition();
        double entropyMonitor();
        double zeroEntryMonitor();

    private:

        LinearOpticalTransform LOCircuit;
        UGeneration UGenerator;
        Eigen::MatrixXcd U;

        void setAntiHermitian( Eigen::MatrixXcd& H,Eigen::VectorXd& position );
        void setPosition(Eigen::MatrixXcd& U, Eigen::VectorXd& position);
        void shiftUToZeroSolution(Eigen::VectorXd& position);

};

#endif // MERITFUNCTION_H_INCLUDED
