CC = g++
CUCC = nvcc
CFLAGS = -Ofast -funroll-loops -c
OBJS = main.o LinearOpticalTransform.o MeritFunction.o BFGS_Optimization.o CUDAFuncs.o

all: LinearOpticalSimulation

LinearOpticalSimulation: $(OBJS)
	$(CUCC) $(OBJS) -o LinearOpticalSimulation

main.o: main.cpp
	$(CC) $(CFLAGS) $(OMPFLAGS) main.cpp

LinearOpticalTransform.o: LinearOpticalTransform.cpp
	$(CC) $(CFLAGS) LinearOpticalTransform.cpp

MeritFunction.o: MeritFunction.cpp
	$(CC) $(CFLAGS) MeritFunction.cpp

BFGS_Optimization.o: BFGS_Optimization.cpp
	$(CC) $(CFLAGS) BFGS_Optimization.cpp

CUDAFuncs.o: CUDAFuncs.cu
	$(CUCC) -c CUDAFuncs.cu

clean:
	rm *.o LinearOpticalSimulation *.dat
