EXEC=spheres
OBJECTS=spheres.o SimpleRNG.o

NVCC=/usr/bin/nvcc
CC=g++
NVCCFLAGS=-Xcompiler -Wall -gencode=arch=compute_20,code=\"sm_20,compute_20\" -m64 --compiler-options -fno-strict-aliasing -I. -I/usr/include -I../common/inc -DUNIX -O2
LDFLAGS=-fPIC -m64 -L/usr/lib64 -L../common/lib/linux -lcudart -L/usr/lib64 -lcublas -lcurand

all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CC) -o $(EXEC) $(OBJECTS) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm *.o $(EXEC)

.PHONY: clean

