#include <omp.h>
#include <iostream>
#include <assert.h>

#define ALLOC alloc_if(1)
#define FREE free_if(1)
#define RETAIN free_if(0)
#define REUSE alloc_if(0)

__declspec(target(mic)) int* array;

class testClass{

    public:

        testClass();
        void runOnMic();

    private:

        void putOnMic();
        void reuseOnMic();
        void globalVarOnMic();
        void reuseGlobalVarOnMic();

        double* a1;

};

testClass::testClass(){

}

void testClass::putOnMic(){

    double testValue;

    /** =======================================

        TO USE CLASS MEMBER VARIABLES POINTERS WITH OMP OFFLOAD,
        WE HAVE TO CREATE A DUMMY POINTER IN THE FUNCTION WHERE WE OFFLOAD THE DATA.

        ANY FURTHER OFFLOADED PORTIONS WITHIN THAT FUNCTION USE THE DUMMY POINTER.

        ANY OFFLOADED PORTIONS OUTSIDE THE SCOPE OF THIS FUNCTION USE THE ORIGINAL POINTER.
        THE NEXT TIME YOU USE THE FUNCTION YOU HAVE TO FREE THE MEMORY, OR THE PROGRAM WILL
        SEG FAULT ON THE MIC.

        FOR FUCKS SAKE.

        JUST USE GLOBAL POINTERS LIKE IN THIS EXAMPLE. NOTE WITH POINTERS OF TYPE __declspec(target(mic))
        THAT YOU STILL NEED TO SPECIFY TO ALLOCATE OR RETAIN THE DATA IT POINTS TO.

        IF YOU USE THE delete[] OPERATION ON THE POINTER - I DON'T THINK IT FREES THE MEMORY ON
        THE MIC. NOT SURE ABOUT THIS THOUGH.

        ======================================= */

    double* dev_a1 = a1;

#pragma offload target(mic) inout( testValue : ALLOC FREE) in(dev_a1[0:100] : ALLOC RETAIN )
#pragma omp parallel
{

    int threadID = omp_get_thread_num();

    if(threadID == 0) testValue = dev_a1[37];

}

    std::cout << testValue << std::endl;

    a1[21] = 777;

    std::cout << dev_a1[99] << std::endl;

#pragma offload target(mic) inout( testValue : ALLOC FREE) nocopy(dev_a1[0:100] : REUSE RETAIN )
#pragma omp parallel
{

    int threadID = omp_get_thread_num();

    if(threadID == 0) testValue = dev_a1[21];

}

    std::cout << testValue << std::endl;

    //free( a1 );

    return;

}

void testClass::reuseOnMic(){

    double testValue;

#pragma offload target(mic) inout( testValue : ALLOC FREE) nocopy(a1[0:100] : REUSE FREE )
#pragma omp parallel
{

    int threadID = omp_get_thread_num();

    if(threadID == 0) testValue = a1[63];

}

    std::cout << testValue << std::endl;

    return;

}


void testClass::globalVarOnMic(){

    int testValue;

    array = new int[1000]; //(int*)malloc(1000 * sizeof(int) );

    for(int i=0;i<1000;i++) array[i] = i;

#pragma offload target(mic) inout( testValue : ALLOC FREE) in(array[0:1000] : ALLOC RETAIN )
#pragma omp parallel
{

    int threadID = omp_get_thread_num();

    if(threadID == 0) testValue = array[436];

}

    std::cout << testValue << std::endl;

    array[436] = -12;

#pragma offload target(mic) inout( testValue : ALLOC FREE) nocopy(array[0:1000] : REUSE RETAIN )
#pragma omp parallel
{

    int threadID = omp_get_thread_num();

    if(threadID == 0) testValue = array[436];

}

    std::cout << testValue << std::endl;

    delete[] array;

    return;

}

void testClass::reuseGlobalVarOnMic(){

    int testValue;

#pragma offload target(mic) inout( testValue : ALLOC FREE) nocopy(array[0:1000] : REUSE RETAIN )
#pragma omp parallel
{

    int threadID = omp_get_thread_num();

    if(threadID == 0) testValue = array[436];

}

    std::cout << testValue << std::endl;

    return;

}

void testClass::runOnMic(){

    a1 = (double*)malloc(100 * sizeof(double) );

    for(int i=0;i<100;i++) a1[i] = i;

    putOnMic();

    reuseOnMic();

    globalVarOnMic();

    for(int i=0;i<20000;i++) reuseGlobalVarOnMic();

    return;

}


int main(){

    testClass pudd;

    pudd.runOnMic();

    return 0;

}
