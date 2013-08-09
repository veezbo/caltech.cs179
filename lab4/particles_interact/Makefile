EXEC=particles_interact
OBJECTS=particles_interact.o interact_kernel.o

NVCC=/usr/bin/nvcc
CC=g++
NVCCFLAGS=-gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\" -m64 --compiler-options -fno-strict-aliasing -I. -I/usr/include -I../common/inc -DUNIX -O2
LDFLAGS=-fPIC -m64 -L/usr/lib64 -L../common/lib/linux -lcudart -lGL -lGLU -lX11 -lXi -lXmu -lGLEW_x86_64 -L/usr/X11R6/lib64 -lGLEW_x86_64 -L/usr/X11R6/lib64 -lglut -L/usr/lib64 -lcutil_x86_64

$(EXEC): $(OBJECTS)
	$(CC) -o $(EXEC) $(OBJECTS) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm *.o $(EXEC)

.PHONY: clean

