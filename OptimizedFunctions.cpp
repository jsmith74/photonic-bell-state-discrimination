#include "OptimizedFunctions.h"

#define ANCILLA_PHOTONS 6
#define ANCILLA_MODES 8
#define HILBERT_SPACE_DIMENSION 75582

OptimizedFunctions::OptimizedFunctions(){


}


void OptimizedFunctions::setReduceGrid(std::vector< std::vector<int> >& nPrime,std::vector< std::vector<int> >& mPrime,int termIntervals,Eigen::MatrixXi& tempReduceGrid){

    tempReduceGrid.resize(0,2);

    int k=0;

    int blockNumb = -1;

    int gridPoint = termIntervals;

    int prevBlockNumb = 0;

    for(int y=0;y<HILBERT_SPACE_DIMENSION;y++){

        do{

            if( k % termIntervals == 0){

                blockNumb++;

            }

            k++;

        } while( std::next_permutation( mPrime[y].begin() , mPrime[y].end() ) );

        if(k >= gridPoint){

            tempReduceGrid.conservativeResize( tempReduceGrid.rows() + 1, 2 );

            tempReduceGrid( tempReduceGrid.rows()-1, 0 ) = prevBlockNumb;
            tempReduceGrid( tempReduceGrid.rows()-1, 1 ) = blockNumb;

            if( k % termIntervals == 0 ) prevBlockNumb = blockNumb + 1;
            else prevBlockNumb = blockNumb;

            gridPoint = ( k - ( k % termIntervals ) + termIntervals );

        }

    }

    if( k < gridPoint ){

        tempReduceGrid.conservativeResize( tempReduceGrid.rows() + 1, 2 );

        tempReduceGrid( tempReduceGrid.rows()-1, 0 ) = prevBlockNumb;
        tempReduceGrid( tempReduceGrid.rows()-1, 1 ) = blockNumb;

    }

    return;

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
