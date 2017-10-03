#include "BFGS_Optimization.h"

#include <fstream>
#include <iostream>
#include <omp.h>

int main(){


    double gradientCheck = 1e-4;

    double maxStepSize = 200.0;

    int intParam = 0;

    BFGS_Optimization optimizer(gradientCheck,maxStepSize,intParam);

    optimizer.minimize();

    return 0;

}
