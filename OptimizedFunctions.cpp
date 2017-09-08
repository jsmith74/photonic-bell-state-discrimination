
#include "OptimizedFunctions.h"


void OptimizedFunctions::setSubNPrimeMPrime(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime,int nPrimeSub[],int mPrimeSub[],int& subIndex,int& totalTermsPerIteration){

    int kn = 0;
    int km = 0;

    int subWall = 0;

    while( subWall < totalTermsPerIteration ){

        do{

            if( subWall >= totalTermsPerIteration ){

                subIndex--;
                break;

            }

            for(int i=0;i<nPrime[ subIndex ].size();i++){

                nPrimeSub[ kn ] = nPrime[ subIndex ][i];

                kn++;

            }

            for(int i=0;i<mPrime[ subIndex ].size();i++){

                mPrimeSub[ km ] = mPrime[ subIndex ][i];

                km++;

            }

            subWall++;

        } while( std::next_permutation( mPrime[ subIndex ].begin(), mPrime[ subIndex ].end() ) );

        subIndex++;

        if( subIndex > nPrime.size() ) break;

    }


    return;

}


OptimizedFunctions::OptimizedFunctions(){



}
