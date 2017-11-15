#include "UGeneration.h"


UGeneration::UGeneration(){



}

void UGeneration::setZeroEntryQuant(Eigen::MatrixXcd& U){

    zeroEntryQuant = 0;

    for(int j=0;j<U.cols();j++) for(int i=0;i<U.rows();i++){

        if(zeroEntries(i,j) == 0) zeroEntryQuant += std::norm( U(i,j) );

    }

    return;

}


void UGeneration::initializeUCondition1(Eigen::MatrixXcd& U){

    int ancillaRows = U.rows() - 4;

    zeroEntries.resize(U.rows(),U.cols());

    initializeZeroEntries();

    setZeroEntriesRandomlyCondition1(ancillaRows);

    while( checkCondition1() ){

        initializeZeroEntries();
        setZeroEntriesRandomlyCondition1(ancillaRows);

    }

    std::cout << "Zero entries of the U matrix:\n" << zeroEntries << std::endl << std::endl;

    return;

}



void UGeneration::setZeroEntriesRandomlyCondition1(int ancillaRows){

    int zeroChoiceRange = ancillaRows + 2;

    for(int j=0;j<zeroEntries.cols();j++){

        int zeroChoices[3];

        zeroChoices[0] = rand() % zeroChoiceRange;

        zeroChoices[1] = rand() % zeroChoiceRange;

        while(zeroChoices[1] == zeroChoices[0]) zeroChoices[1] = rand() % zeroChoiceRange;

        zeroChoices[2] = rand() % zeroChoiceRange;

        while(zeroChoices[2] == zeroChoices[1] || zeroChoices[2] == zeroChoices[0]) zeroChoices[2] = rand() % zeroChoiceRange;

        for(int i=0;i<3;i++){

            if(zeroChoices[i] < ancillaRows) zeroEntries(zeroChoices[i],j) = 0;
            else if(zeroChoices[i]==ancillaRows){
                zeroEntries(ancillaRows,j) = 0;
                zeroEntries(ancillaRows+1,j) = 0;
            }
            else if(zeroChoices[i]==ancillaRows+1){
                zeroEntries(ancillaRows+2,j) = 0;
                zeroEntries(ancillaRows+3,j) = 0;
            }

        }
    }

}

void UGeneration::initializeZeroEntries(){

    for(int i=0;i<zeroEntries.rows();i++) for(int j=0;j<zeroEntries.cols();j++) zeroEntries(i,j) = 1;

    return;

}

bool UGeneration::checkCondition1(){

    for(int j=0;j<zeroEntries.rows();j++){

        if( zeroEntries.row(j).sum() == 0 ) return true;

    }

    for(int j=0;j<zeroEntries.cols();j++){

        Eigen::VectorXi zeroChoices;

        zeroChoices.resize(0);

        int k=0;
        for(int i=0;i<zeroEntries.rows();i++){
            if(zeroEntries(i,j) == 0){
                zeroChoices.conservativeResize( k+1 );
                zeroChoices[k] = i;
                k++;
            }
        }

        for(int i=j+1;i<zeroEntries.cols();i++){

            bool weGood = false;

            for(int l=0;l<zeroChoices.size();l++) if(zeroEntries( zeroChoices(l),i ) == 0) weGood = true;

            if(!weGood) return true;

        }

    }

    return false;

}
