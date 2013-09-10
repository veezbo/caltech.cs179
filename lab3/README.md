CS179 Lab 3
----------

Hopefully I've supplied enough explanation here, but in case I haven't feel free to ask me.


In main.cpp, the only changes I made were to the initPhysics method where I set the initial positions to a circle (or rather a ring) with radius 1, and thickness 0.05 centered at (0.5, 0.5). The points start with uniform random lives between 0 and 1. This configuration leads to interesting symmetries.


In physics.frag, I set various constants that are involved in the physics calculations. Essentially, if the life is less than 0, then I reset the point to somewhere on the original circle. Otherwise, I apply the gravitational force, which varies with 1/r^2 within a certain radius, and is 5000 times as much outside this radius (this keeps the points in view). Then, I use the explicit Euler integration scheme to calculate velocity, and then clip it to some maximum velocity in each direction. Finally, I use the explicit Euler again to calculate position. For each particle, the life is decreased by some set amount at each iteration.


In energy.frag, I make the standard gravitational potential energy and kinetic energy calculations, assuming the mass of the particles is 1. However, since the gravitational force is 5000 times as large outside a particular radius, the potential energy is calculated as:

    5000 * -1/(radius - smallerGravityRadius) + -1/(smallerGravityRadius)   for radius > smallerGravityRadius
    -1/radius 															    for radius < smallerGravityRadius

and kinetic energy is calculated by:

    0.5 * norm(velocity).


In reduction.frag, the calculations are standard. I simply sample the reduction texture at the given texture coordinates and additionally at the 3 surrounding coordinates, and calculate the average. However, since this average is set to the color of the particles, I take the absolute value since the energy is sometimes negative (and negative colors do not seem to work on newer cards). I've adjusted the "Average Energy" label accordingly.