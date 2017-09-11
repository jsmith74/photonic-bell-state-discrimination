#ifndef OPTIMIZEDFUNCTIONS_H_INCLUDED
#define OPTIMIZEDFUNCTIONS_H_INCLUDED

#include <vector>
#include <algorithm>


class OptimizedFunctions{

    public:

        OptimizedFunctions();
        void initializeStartingNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime,int nPrimeStarter[],int mPrimeStarter[],int numberOfThreads,int termIntervals);


    private:



};

#endif // OPTIMIZEDFUNCTIONS_H_INCLUDED
