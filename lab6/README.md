CS179 Lab 6
---------

## PROGRAM REQUIREMENTS:

1) numSpheres should be no larger than blockDim (the number of threads per block). Currently, this number is set to 512 because that's the max allowed on current GPU's.
2) numPts should be divisible by blockDim (the number of threads per block), and that is currently 512.

## COMPILING INSTRUCTIONS:

Simply make in the appropriate directory, and run the generated executable. Everything was initially compiled on both minuteman and mx in the ANB lab.

## PROGRAM DETAILS:

* Find Volume (CPU):
  - I added the line that checks whether a point is in a sphere to complete the CPU code.

* Find Volume (GPU + CUBLAS):
  - I implemented the CheckPointsK kernel, making sure to carefully distinguish between doubleResults (for CUBLAS) and intResults (for reduction), and also completed the volumeCUBLAS method with appropriate memory management.

* Find Volume (GPU + Reduction):
  - I implemented the GenerateRandom3K kernel using curand, as well as a somewhat optimized reduction kernel. I used both of these in the volumeCUDA method with approriate memory management and handling of the reduction calls.


