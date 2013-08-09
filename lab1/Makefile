OBJECTS=main.o ArcBall.o glFunctions.o Mesh.o readpng.o uistate.o Vector3.o
SOURCES=*.ccp *.h
CFLAGS=-g
LDFLAGS= -L/usr/lib/nvidia -L/usr/X11R6/lib -lglut -lGL -lGLU -lpng #-lXi -lXmu

all: glslRenderer

glslRenderer: $(OBJECTS)
	g++ $(LDFLAGS) -o glslRenderer $^

.cpp.o:
	g++ $(CFLAGS) -c $<

clean:
	-rm $(OBJECTS) glslRenderer
