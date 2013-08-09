OBJECTS=main.o ArcBall.o glFunctions.o readpng.o uistate.o Vector3.o
SOURCES=*.ccp *.h
CFLAGS=-g
LDFLAGS= -L/usr/local/lib -L/usr/nvidia/lib -L/usr/X11R6/lib -lglut -lGL -lGLU -lpng #-lXi -lXmu

all: glslRenderer

glslRenderer: $(OBJECTS)
	g++ $(LDFLAGS) -o glslRenderer $^

.cpp.o:
	g++ $(CFLAGS) -c $<

clean:
	-rm $(OBJECTS) glslRenderer
