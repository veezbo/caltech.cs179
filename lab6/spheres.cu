// Lab 6 - volume of union of spheres

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cublas.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cutil_math.h>

#include "SimpleRNG.h"

// spheres represented with x, y, z and posn and w as radius

// macro for error-checking CUDA calls
#define CUDA_SAFE_CALL(x) do { cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    printf("Error %d \"%s\" at %s:%u\n", err, cudaGetErrorString(err), \
        __FILE__, __LINE__); \
    exit(-1); } } while (false)

////////////////////////////////////////////////////////////////////////////////
// helper functions for this lab...

typedef struct {
  double xmin, xmax, ymin, ymax, zmin, zmax;
  double xrange, yrange, zrange;
  double volume;
} BoundBox;

// find the bounding box for a set of spheres
void FindBoundingBox(float4* spheres, int numSpheres, BoundBox& box) {
  box.xmin = box.xmax = spheres[0].x;
  box.ymin = box.ymax = spheres[0].y;
  box.zmin = box.zmax = spheres[0].z;
  for (int x = 0; x < numSpheres; x++) {
    if (box.xmin > spheres[x].x - spheres[x].w)
      box.xmin = spheres[x].x - spheres[x].w;
    if (box.ymin > spheres[x].y - spheres[x].w)
      box.ymin = spheres[x].y - spheres[x].w;
    if (box.zmin > spheres[x].z - spheres[x].w)
      box.zmin = spheres[x].z - spheres[x].w;
    if (box.xmax < spheres[x].x + spheres[x].w)
      box.xmax = spheres[x].x + spheres[x].w;
    if (box.ymax < spheres[x].y + spheres[x].w)
      box.ymax = spheres[x].y + spheres[x].w;
    if (box.zmax < spheres[x].z + spheres[x].w)
      box.zmax = spheres[x].z + spheres[x].w;
  }
  box.xrange = box.xmax - box.xmin;
  box.yrange = box.ymax - box.ymin;
  box.zrange = box.zmax - box.zmin;
  box.volume = box.xrange * box.yrange * box.zrange;
}

// return the current time, in seconds
double now() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

// generate a "random" seed based on time
long long random_seed() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec;
}

// check if a point is inside a sphere
__device__ __host__ bool PointInSphere(float3& pt, float4& sphere) {
  return length(pt - make_float3(sphere)) < sphere.w;
}

SimpleRNG rng;

////////////////////////////////////////////////////////////////////////////////
// kernels

// inputs:
//   spheres, numSpheres - describe the array of spheres
//   points - points to check against spheres; coordinates are in [0, 1]^3
//   doubleResults, intResults - arrays of doubles and floats to write results
//     to. either can be NULL, in which case results aren't written to them
//   box - bounding box to scale points into
// total number of threads must be equal to the number of points
__global__ void CheckPointsK(float4* spheres, int numSpheres, float3* points,
    double* doubleResults, unsigned int* intResults, BoundBox box) {

  // : check if the point is inside any sphere. if so, set the appropriate
  // entry in doubleResults and intResults to 1 (if non-NULL).

  //Decide whether we want to write to doubleResults or intResults
  int doubleRun;
  if (doubleResults == NULL)
  {
    doubleRun = 0;
  }
  else
  {
    doubleRun = 1;
  }

  extern __shared__ float4 sphereLocs[];
  int blockId = blockIdx.x;
  int blockD = blockDim.x;
  int threadId = threadIdx.x;
  int globalId = blockD*blockId + threadId;

  //copy the global spheres to shared memory, but only using as many threads as we need to (i.e. numSpheres many)
  //: Require that numSpheres be no larger than numThreads (per block)
  if (threadId <= numSpheres)
  {
    sphereLocs[threadId] = spheres[threadId];
  }
  __syncthreads();

  //read in the point once from global memory (using its global id) and loop over all the spheres to see if in the collective volume
  float3 oldPoint = points[globalId];
  float3 point = make_float3(oldPoint.x*box.xrange+box.xmin, oldPoint.y*box.yrange+box.ymin, oldPoint.z*box.zrange+box.zmin);

  //Compare doubleRun outside of the loop so we only have do it once
  if (doubleRun == 1)
  {
    doubleResults[globalId] = 0.;
    for (int j = 0; j < numSpheres; j++)
    {
      if (PointInSphere(point, sphereLocs[j]))
      {
        doubleResults[globalId] = 1.;
        break;
      }
    }
  }
  else
  {
    intResults[globalId] = 0;
    for (int j = 0; j < numSpheres; j++)
    {
      if (PointInSphere(point, sphereLocs[j]))
      {
        intResults[globalId] = 1;
        break;
      }
    }
  }

}

// generates 'count' random float3s using CURAND
// only requires the total number of threads to be a factor of 'count'
// ex. can call as such: GenerateRandom3K<<< 3, 8 >>>(..., 72)
__global__ void GenerateRandom3K(float3* toWrite, long long seed,
                                 curandState* states, int count) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;

  // : initialize random generator states, then generate random float3s in
  // [0, 1]^3
  float a, b, c;
  //curandState *localState = &states[0];
  curand_init(seed, index, 0, &states[index]);
  //curand_init(seed, 0, 0, states);

  for (int x = index; x < count; x += blockDim.x * gridDim.x)
  {
    a = curand_uniform(&states[index]);
    b = curand_uniform(&states[index]);
    c = curand_uniform(&states[index]);

    toWrite[x] = make_float3(a, b, c);
  }
    
}

// : add a reduction kernel to sum an array of unsigned ints, for VolumeCUDA

__global__ void reduce(unsigned int *g_idata) {
  extern __shared__ int sdata[];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s = blockDim.x/2; s > 0; s>>=1) {
    if (tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_idata[blockIdx.x] = sdata[0];
}


////////////////////////////////////////////////////////////////////////////////
// host code

// find volume on CPU
double VolumeCPU(float4* spheres, int numSpheres, int numPts, BoundBox& box) {

  int x, y, numPtsInside = 0;
  for (x = 0; x < numPts; x++) {
    float3 pt = make_float3(rng.GetUniform() * box.xrange + box.xmin,
                            rng.GetUniform() * box.yrange + box.ymin,
                            rng.GetUniform() * box.zrange + box.zmin);
    for (y = 0; y < numSpheres; y++)
      if (PointInSphere(pt, spheres[y]))
        break;
    if (y < numSpheres)
      numPtsInside++;
  }

  return (double)numPtsInside / (double)numPts * box.volume;
}

// find volume on GPU, summing using CUBLAS
double VolumeCUBLAS(float4* d_spheres, int numSpheres, int numPts,
                   BoundBox& box) {

  //double vol = 0.0;
  const int numThreads = 512; //: Indicate on README that we want numPts divisible by 512
  dim3 grid (numPts / numThreads, 1, 1);
  dim3 block (numThreads, 1, 1);

  // :
  // 1. allocate memory for needed data
  // 2. generate random points on GPU in [0, 1]^3 using CURAND host API
  // 3. check if each point is within any sphere
  // 4. count points using CUBLAS
  // 5. free memory on GPU

  //STEP 1
  unsigned int spheresSize = numSpheres * sizeof(float4);
  unsigned int pointsSize = numPts * sizeof(float3);
  unsigned int doubleResultsSize = numPts * sizeof(double);
  float3* points;
  double* doubleResults;
  unsigned int* intResults = 0;

  CUDA_SAFE_CALL(cudaMalloc((void**)&points, pointsSize));
  CUDA_SAFE_CALL(cudaMalloc((void**)&doubleResults, doubleResultsSize));

  //STEP 2
  curandGenerator_t r;
  curandCreateGenerator(&r, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(r, (long long)random_seed);
  curandGenerateUniform(r, (float*)points, numPts*3);
  curandDestroyGenerator(r);

  //STEP 3

  CheckPointsK<<< grid, block, spheresSize >>> (d_spheres, numSpheres, points,
          doubleResults, intResults, box);

  //STEP 4
  double result;
  result = cublasDasum(numPts, doubleResults, 1);

  //STEP 5
  CUDA_SAFE_CALL(cudaFree(points));
  CUDA_SAFE_CALL(cudaFree(doubleResults));

  return (result / numPts) * (double) box.volume;
}

// find volume on GPU, summing using reduction kernel
double VolumeCUDA(float4* d_spheres, int numSpheres, int numPts, BoundBox& box) {

  //double vol = 0.0;
  const int numThreads = 512; //: Indicate on README that we want numPts divisible by 512
  dim3 grid (numPts / numThreads, 1, 1);
  dim3 block (numThreads, 1, 1);

  // :
  // 1. allocate memory for needed data (including random generator states)
  // 2. generate random points on GPU in [0, 1]^3 using CURAND device API
  // 3. check if each point is within any sphere
  // 4. count points using reduction kernel
  // 5. free memory on GPU

  //STEP 1
  unsigned int spheresSize = numSpheres * sizeof(float4);
  unsigned int pointsSize = numPts * sizeof(float3);
  unsigned int intResultsSize = numPts * sizeof(int);
  float3* points;
  double* doubleResults = 0;
  unsigned int* intResults;
  curandState* devStates;

  CUDA_SAFE_CALL(cudaMalloc((void**)&devStates, numThreads * sizeof(curandState)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&points, pointsSize));
  
  //STEP 2
  dim3 randomGrid (1, 1, 1);
  dim3 randomBlock (numThreads, 1, 1);
  GenerateRandom3K<<< randomGrid, randomBlock >>> (points, random_seed(), devStates, numPts);

  CUDA_SAFE_CALL(cudaFree(devStates));

  //STEP 3
  CUDA_SAFE_CALL(cudaMalloc((void**)&intResults, intResultsSize));

  CheckPointsK<<< grid, block, spheresSize >>> (d_spheres, numSpheres, points,
           doubleResults, intResults, box);

  CUDA_SAFE_CALL(cudaFree(points));

  //STEP 4
  int numReduceThreads = 512; //the factor by which to reduce the number of points by
  int dim;
  for (dim = numPts; dim >= numReduceThreads; dim/= numReduceThreads)
  { 
    dim3 reductionBlock (numReduceThreads, 1, 1);
    dim3 reductionGrid  (dim / numReduceThreads, 1, 1);
    unsigned int sharedMemSize = numReduceThreads*sizeof(int);

    reduce<<< reductionGrid, reductionBlock, sharedMemSize >>> (intResults);
  }

  numReduceThreads = dim;
  dim3 reductionBlock (numReduceThreads, 1, 1);
  dim3 reductionGrid (dim / numReduceThreads, 1, 1);
  unsigned int sharedMemSize = numReduceThreads*sizeof(int);

  reduce<<< reductionGrid, reductionBlock, sharedMemSize >>> (intResults);

  //STEP 5
  unsigned int result;
  CUDA_SAFE_CALL(cudaMemcpy(&result, &intResults[0], sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(intResults));

  return ((double)result / (double)numPts) * (double) box.volume;

}

////////////////////////////////////////////////////////////////////////////////
// main program

void RunVolume(const char* name, double (*Vol)(float4*, int, int, BoundBox&),
    float4* spheres, int numSpheres, int numPts, BoundBox& box) {
  printf("find volume (%s)...\n", name);
  double start_time = now();
  double volume = Vol(spheres, numSpheres, numPts, box);
  double end_time = now();
  printf("  volume: %g\n", volume);
  printf("  time: %g sec\n", end_time - start_time);
}

int main(int argc, char** argv) {

  // seed the CPU random generator
  rng.SetState(random_seed(), random_seed());

  // set program parameters and allocate memory for spheres
  printf("generate spheres...\n");
  int numPts = 1024 * 1024 * 16;
  int numSpheres = 100;
  float4* spheres = (float4*)malloc(numPts * sizeof(float4));
  if (!spheres) {
    printf("failed to allocate memory for spheres\n");
    return -1;
  }

  // generate random spheres centered in [0, 10]^3
  double totalVolume = 0.0f;
  for (int x = 0; x < numSpheres; x++) {
    spheres[x].x = rng.GetUniform() * 10.0f;
    spheres[x].y = rng.GetUniform() * 10.0f;
    spheres[x].z = rng.GetUniform() * 10.0f;
    spheres[x].w = rng.GetUniform() + 1.0f;
    totalVolume += (4.0f * spheres[x].w * spheres[x].w * spheres[x].w * M_PI
                    / 3.0f);
    // uncomment to print spheres
    //printf("  sphere: (%g, %g, %g) with r = %g\n", spheres[x].x, spheres[x].y,
    //       spheres[x].z, spheres[x].w);
  }
  printf("  number of spheres: %u\n", numSpheres);
  printf("  non-union volume: %g\n", totalVolume);
  printf("  number of points: %u\n", numPts);

  // find bounding box of spheres
  printf("find bounds rect...\n");
  BoundBox box;
  FindBoundingBox(spheres, numSpheres, box);
  printf("  boundsrect: [%g, %g] x [%g, %g] x [%g, %g]\n", box.xmin, box.xmax,
         box.ymin, box.ymax, box.zmin, box.zmax);
  printf("  boundsrange: %g, %g, %g (volume %g)\n", box.xrange, box.yrange,
         box.zrange, box.volume);

  // init cublas and allocate memory on the GPU
  printf("initialize GPU...\n");
  cublasInit();
  float4* d_spheres;
  CUDA_SAFE_CALL(cudaMalloc(&d_spheres, numSpheres * sizeof(float4)));

  // copy the spheres to the GPU
  cudaMemcpy(d_spheres, spheres, numSpheres * sizeof(float4),
             cudaMemcpyHostToDevice);

  // run CPU version
  RunVolume("CPU", VolumeCPU, spheres, numSpheres, numPts, box);
  RunVolume("CUBLAS", VolumeCUBLAS, d_spheres, numSpheres, numPts, box);
  RunVolume("no CUBLAS", VolumeCUDA, d_spheres, numSpheres, numPts, box);

  // get rid of stuff in memory
  printf("clean up...\n");
  CUDA_SAFE_CALL(cudaFree(d_spheres));
  cublasShutdown();

  cudaThreadExit();
  return 0;
}

