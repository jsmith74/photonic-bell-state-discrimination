#include "OptimizedFunctions.h"

#define ANCILLA_PHOTONS 6
#define ANCILLA_MODES 8
#define HILBERT_SPACE_DIMENSION 75582

OptimizedFunctions::OptimizedFunctions(){


}


void OptimizedFunctions::initializeStartingNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime,int nPrimeStarter[],int mPrimeStarter[],int numberOfThreads,int termIntervals){

    int k = 0;
    int j = 0;

    for(int y=0;y<HILBERT_SPACE_DIMENSION;y++){

        do{

            if( k % termIntervals == 0 ){

                for(int i=0;i<nPrime[y].size();i++) nPrimeStarter[ i + j*(4 + ANCILLA_MODES) ] = nPrime[y][i];
                for(int i=0;i<mPrime[y].size();i++) mPrimeStarter[ i + j*(2 + ANCILLA_PHOTONS) ] = mPrime[y][i];

                j++;

            }

            k++;

        } while( std::next_permutation( mPrime[y].begin() , mPrime[y].end() ) );

    }

    return;

}
