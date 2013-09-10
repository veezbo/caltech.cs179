CS179 Lab 5
--------------

Everything ran and compiled on Ubuntu 13.04 using a GTX 460. Using this setup gave me both very fast arcball transformations, and the included epsilon/julia set change calculations (since this requires recalculation of the whole Julia set).
To compile, simply run ```make``` (though you may need to use the common folder as well).

### SUMMARY OF CHANGES:

In the host code:
* I did nothing but change whatever TODO's were necessary to get the code running.

In the device code:
* JuliaDist 			: Implemented as described on the recitation slides- nothing special here.
* JuliaNormal			: Called JuliaDist on surrounding positions in 3D to find the gradient.
* render (kernel) 		:	
    - Created a light source from which I do diffuse color calculation.
    - Used the standard ray-tracing algorithm to find the first intersecting position.
        * However, I adjusted the epsilon linearly with how close the position was to the camera.
        * Additionally, I adjusted the time step linearly with the JuliaDist of the current position.
	    * Together, the above 2 gave me a huge speed-up (compared to without their implementations) when doing any of the possible transformations.
    - Calculated color based on ambient and diffuse coloring.
        * Ambient color is calculated by position ( in [0, 1] ).
        * Diffuse color is calculated with the standard NdotL factor, multiplied by the base position color.
* setfractal (kernel) 	: Done in the standard way as dictated on the recitation slides. Simply calculating JuliaDist for each of the volume points.


#### MEMORY COALESCING:

    Using memory coalescing gives these times for 10 'q' presses:
	  Recompute took 38.719521 ms
	  Recompute took 36.513248 ms
	  Recompute took 34.891872 ms
	  Recompute took 32.831650 ms
	  Recompute took 32.629791 ms
	  Recompute took 32.356544 ms
	  Recompute took 32.802654 ms
	  Recompute took 31.734465 ms
	  Recompute took 31.344383 ms
	  Recompute took 30.926975 ms

#### NON-COALESCING:
	
    Not using coalescing gives these times for 10 'q' presses:
	  Recompute took 39.311550 ms
	  Recompute took 36.476479 ms
	  Recompute took 35.465088 ms
	  Recompute took 33.841793 ms
	  Recompute took 31.779903 ms
	  Recompute took 31.102943 ms
	  Recompute took 28.957567 ms
	  Recompute took 28.736256 ms
	  Recompute took 27.675200 ms
	  Recompute took 27.505119 ms

This is weirdly faster towards the ending clicks, however the display looks fairly bad with anomalies everywhere (and, I suspect, it's only faster towards ending clicks because it renders fewer points properly).


### EXTRA CREDIT:

I implemented the adjusted epsilon factor linearly with closeness to he camera (i.e. the eye Positions of our rays). This, along with the adjusted time step, resulted in a significant speedup. 
