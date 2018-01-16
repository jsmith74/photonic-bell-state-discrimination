#include <fstream>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <Eigen/Dense>
#include "UGeneration.h"
#include "LinearOpticalTransform.h"

#define CONDITIONED_U

int main(){

    int ancillaPhotons = 6;
    int ancillaModes = 6;

    int trialNumb = 1;

    srand(611*time(NULL));

    LinearOpticalTransform LOCircuit;

    LOCircuit.initializeCircuit(ancillaPhotons,ancillaModes);

    while(trialNumb <= 1237){

        Eigen::MatrixXcd U(4 + ancillaModes,4 + ancillaModes);

        UGeneration UGenerator;

        UGenerator.initializeUCondition1(U);

        LOCircuit.setMutualEntropy(U);

        #ifdef CONDITIONED_U

            std::ofstream outfile("conditionedU.dat",std::ofstream::app);

        #endif // CONDITIONED_U

        #ifndef CONDITIONED_U

            std::ofstream outfile("UnconditionedU.dat",std::ofstream::app);

        #endif // CONDITIONED_U

        outfile << trialNumb << "\t" << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualEntropy << std::endl;

        outfile.close();

        trialNumb++;

    }

    return 0;

}
