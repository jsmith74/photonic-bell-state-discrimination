#include "BFGS_Optimization.h"

#include <fstream>
#include <iostream>
#include <omp.h>

int main(){

    int CPUWorkload = 188400000;

    double gradientCheck = 1e-4;

    double maxStepSize = 200.0;

    BFGS_Optimization optimizer(gradientCheck,maxStepSize,CPUWorkload);

    for(int i=0;i<1000;i++) optimizer.minimize();

    return 0;

}
