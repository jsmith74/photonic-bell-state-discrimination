CC = icpc
CFLAGS = -O3 -xavx -c
LFLAGS = -O3 -xavx
EIGENFLAGS = -inline-forceinline -funroll-loops
OBJS = main.o LinearOpticalTransform.o MeritFunction.o UGeneration.o BFGS_Optimization.o
OMPFLAGS = -fopenmp

all: LinearOpticalSimulation

LinearOpticalSimulation: $(OBJS)
	$(CC) $(LFLAGS) $(OMPFLAGS) $(EIGENFLAGS) $(OBJS) -o LinearOpticalSimulation

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

LinearOpticalTransform.o: LinearOpticalTransform.cpp
	$(CC) $(CFLAGS) $(OMPFLAGS) LinearOpticalTransform.cpp

MeritFunction.o: MeritFunction.cpp
	$(CC) $(CFLAGS) $(EIGENFLAGS) MeritFunction.cpp

BFGS_Optimization.o: BFGS_Optimization.cpp
	$(CC) $(CFLAGS) $(EIGENFLAGS) BFGS_Optimization.cpp

UGeneration.o: UGeneration.cpp
	$(CC) $(CFLAGS) $(EIGENFLAGS) UGeneration.cpp

clean:
	rm *.o LinearOpticalSimulation *.dat *.out
