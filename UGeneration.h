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
        void findZeroUnitary(Eigen::VectorXd& position);

        double zeroEntryQuant;
        Eigen::MatrixXi zeroEntries;

    private:

        void initializeZeroEntries();
        void setZeroEntriesRandomlyCondition1(int ancillaRows);
        bool checkCondition1();

        void setAntiHermitian( Eigen::MatrixXcd& H,Eigen::VectorXd& position );
        void minimize(Eigen::VectorXd& position);
        void setGradient(Eigen::VectorXd& position,Eigen::VectorXd& gradient);
        double alpha(Eigen::VectorXd& position,Eigen::VectorXd& gradient,Eigen::VectorXd& p);
        void setAlphaJ(double& alphaj,double& alphaLow,double& alphaHigh,double& phiLow,double& phiHigh,double& phiLowPrime);
        double zoom(Eigen::VectorXd& position,double alphaLow,double alphaHigh,double phiLow,double phiHigh,double phiLowPrime);
        double phiPrime(Eigen::VectorXd& position,double& a);
        double phi(Eigen::VectorXd& position,double& a);
        double f( Eigen::VectorXd& position );

        double stepMonitor, alphaMax, secondDerivativeTest;
        double phiPrime0;
        double phi0;
        Eigen::VectorXd alphaPosition;
        Eigen::VectorXd p;

};

#endif // UGENERATION_H_INCLUDED
