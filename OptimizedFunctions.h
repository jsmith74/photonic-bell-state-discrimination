#ifndef OPTIMIZEDFUNCTIONS_H_INCLUDED
#define OPTIMIZEDFUNCTIONS_H_INCLUDED

#include <vector>
#include <algorithm>
#include <iostream>

class OptimizedFunctions{

    public:

        OptimizedFunctions();
        void setSubNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime,int nPrimeSub[],int mPrimeSub[],int& subIndex,int& totalTermsPerIteration);

    private:


};

#endif // OPTIMIZEDFUNCTIONS_H_INCLUDED
