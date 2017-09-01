#include "BFGS_Optimization.h"

#include <fstream>
#include <iostream>
#include <omp.h>
#include <unistd.h>

int main(){

#pragma omp parallel for schedule(dynamic) default(none)
    for(int i=0;i<1000;i++){

        if(i<200) usleep(2000000 * omp_get_thread_num());

        double gradientCheck = 1e-4;

        double maxStepSize = 200.0;

        int intParam = 0;

        BFGS_Optimization optimizer(gradientCheck,maxStepSize,intParam);

        optimizer.minimize();

    }

    return 0;

}
