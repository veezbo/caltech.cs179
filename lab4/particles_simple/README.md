CS179 Lab 4: particles_simple
------------------

The only changes made to the original code were:

* *particles_simple.cu*: Changed the initial positions in temppos to positions all across the screen (i.e. in 4 2D quadrants, rather than just the first).

* *simple_kernel.cu*: Simply added the position update based on initial velocity for all the points- no calculation of acceleration.


Everything ran and compiled on Ubuntu 13.04 using a GTX 460.