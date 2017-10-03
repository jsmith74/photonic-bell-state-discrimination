#ifndef MERITFUNCTION_H_INCLUDED
#define MERITFUNCTION_H_INCLUDED

#include "LinearOpticalTransform.h"

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

    private:

        LinearOpticalTransform LOCircuit;
        Eigen::MatrixXcd U,V,W;
        Eigen::VectorXd D;

        void setAntiHermitian1( Eigen::MatrixXcd& H,Eigen::VectorXd& position );
        void setAntiHermitian2( Eigen::MatrixXcd& H,Eigen::VectorXd& position );

        void setPosition1( Eigen::VectorXd& position );
        void setPosition2( Eigen::VectorXd& position );

};

#endif // MERITFUNCTION_H_INCLUDED
