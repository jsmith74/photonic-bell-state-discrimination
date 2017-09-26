#include "BFGS_Optimization.h"

#include <fstream>
#include <iostream>
#include <omp.h>

int main( int argc, char *argv[] ){

    int CPUWorkload = std::atoi( argv[1] );

    for(int i=0;i<1000;i++){

        double gradientCheck = 1e-4;

        double maxStepSize = 200.0;

        BFGS_Optimization optimizer(gradientCheck,maxStepSize,CPUWorkload);

        optimizer.minimize();

    }

    return 0;

}
