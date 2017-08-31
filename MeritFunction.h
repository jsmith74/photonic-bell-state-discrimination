#ifndef MERITFUNCTION_H_INCLUDED
#define MERITFUNCTION_H_INCLUDED

#include "LinearOpticalTransform.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unsupported/Eigen/MatrixFunctions>

class MeritFunction{

    public:

        MeritFunction();
        void setMeritFunction(int intParam);
        double f(Eigen::VectorXd& position);
        int funcDimension;
        void printReport(Eigen::VectorXd& position);
        Eigen::VectorXd setInitialPosition();

    private:

        LinearOpticalTransform LOCircuit;
        Eigen::MatrixXcd U;

        void checkSVDInitialConditionScaling();

};

#endif // MERITFUNCTION_H_INCLUDED
