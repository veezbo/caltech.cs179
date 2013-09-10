CS179 Lab 4: particles_interact
----------------

## Description of Code:

### particles_interact.cu:
	Added the necessary mapping and unmapping of OpenGL buffers, initialization and calling of the kernel, and also pingponging.
	Set the initial positions to the 4 quadrants in 2D, with random positions centered around points that are reasonably close.

### interact_kernel.cu: 
	Implements basic separation and cohesion flocking with the naive O(n^2) algorithm. We store the positions of particles in just one particular block at a time in shared memory, and sum up the position and calculate the acceleration due to those particles all at once before reading in the next block. 
	After doing this for all blocks, we have a total acceleration due to all particles (although only those within a radius of NEIGHBOUR_DIST are counted) for both separation and cohesion.
	We then use symplectic Euler integration to calculate the new velocity (which we normalize) and position.


**NECESSARY RUNNING CONDITIONS**: ```BLOCK_SIZE``` must divide ```numBodies```

**PREFERRED RUNNING CONDITIONS**: Try running with 6400 particles, and the current constant values.  

Everything ran and compiled on Ubuntu 13.04 using a GTX 460.

*As a side note, I was able to get cuda print debugging to work, but it's unnecessary for the completed project, so I commented out any inclusions in the code*.