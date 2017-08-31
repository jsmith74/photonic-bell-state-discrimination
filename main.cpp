#include "BFGS_Optimization.h"

#include <time.h>
#include <fstream>
#include <iostream>

int main(){

    double gradientCheck = 1e-6;

    double maxStepSize = 2.0;

    int intParam = 0;

    clock_t t1,t2;

    BFGS_Optimization optimizer(gradientCheck,maxStepSize,intParam);

    t1 = clock();
    for(int i=0;i<1000;i++) optimizer.minimize();
    t2 = clock();

    float diff = (float)t2 - (float)t1;

    std::cout << "Runtime: " << diff/CLOCKS_PER_SEC << std::endl << std::endl;

    // TO DO: review SVD Non-unitary matrix generators and write it. Don't _always_ re-normalize the singular values
    // just do it when when the matrix becomes super unitary. We want to allow the singular values to be <=1 and
    // re-normalizing in this regime will force at least one singular value to always be 1.

    // Think about the optimization, think about an appropriate starting regime for the optimization.
    // Parallelize this, also think about implementing CUDA again with this code if timing becomes an issue.

    return 0;

}
