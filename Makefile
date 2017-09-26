CC = icpc
CFLAGS = -O3 -xavx -c
LFLAGS = -O3 -xavx
OBJS = main.o LinearOpticalTransform.o MeritFunction.o BFGS_Optimization.o
OMPFLAGS = -fopenmp

all: LinearOpticalSimulation Script

LinearOpticalSimulation: $(OBJS)
	$(CC) $(LFLAGS) $(OMPFLAGS) $(OBJS) -o LinearOpticalSimulation

Script: script.cpp
	$(CC) script.cpp -o Script

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

LinearOpticalTransform.o: LinearOpticalTransform.cpp
	$(CC) $(CFLAGS) $(OMPFLAGS) LinearOpticalTransform.cpp

MeritFunction.o: MeritFunction.cpp
	$(CC) $(CFLAGS) MeritFunction.cpp

BFGS_Optimization.o: BFGS_Optimization.cpp
	$(CC) $(CFLAGS) BFGS_Optimization.cpp

clean:
	rm *.o LinearOpticalSimulation *.dat *.out Script
