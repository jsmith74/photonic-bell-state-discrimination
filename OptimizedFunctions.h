#ifndef OPTIMIZEDFUNCTIONS_H_INCLUDED
#define OPTIMIZEDFUNCTIONS_H_INCLUDED

#include <vector>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <Eigen/Dense>

class OptimizedFunctions{

    public:

        OptimizedFunctions();
        void initializeStartingNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime,int nPrimeStarter[],int mPrimeStarter[],int numberOfThreads,int termIntervals);
        void setReduceGrid(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime,int termIntervals,Eigen::MatrixXi& tempReduceGrid);

    private:



};

#endif // OPTIMIZEDFUNCTIONS_H_INCLUDED
