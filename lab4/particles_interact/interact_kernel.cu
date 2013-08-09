//Basic Math Include
#include <cutil_math.h>

//print include- for debugging purposes
//#include "cuPrintf.cu"

#define WRAP(x,m) (((x)<(m))?(x):((x)-(m)))
#define VEC3(v) (make_float3((v).x, (v).y, (v).z))
#define UNIT(v) ((v) / (dot(make_float3((v).x, (v).y, (v).z), make_float3((v).x, (v).y, (v).z)) + softeningFactor))

#define FORCE_CONSTANT 50.0f
#define FLOCK_CONSTANT 250.0f
#define VELOCITY_CONSTANT 3000.0f

#define NEIGHBOR_DIST 5.0f
#define FLOCKING_DIST 2.5f

#define softeningFactor 0.01f

//Calculates the acceleration due to one particle-particle interaction (the reason the naive algorithm is O(n^2))
__device__ float3 particle_particle( float4 bi, float4 bj)
{
	float4 radius = bi - bj;
	float dist = length(radius);
	if (dist > NEIGHBOR_DIST)
	{
		return make_float3(0., 0., 0.);
	}
	float distanceSquared = dist * dist;
	distanceSquared += softeningFactor;

	float distanceSixth = distanceSquared * distanceSquared * distanceSquared;
	float distanceCubed = sqrt(distanceSixth);

	// bi.w and bi.j are the masses, both currently initialized to 1
	float forceApplied = (bi.w * bj.w) / distanceCubed;

	float4 accl = UNIT(radius) * forceApplied;

	return VEC3(accl);
}

//Calculates the net acceleration on a particular particle of all particles (each handled by one thread) in a block
__device__ float3 compute_accel(float4 pos, float4* curPos, int numBodies)
{
    float3 accl = make_float3(0., 0., 0.);

    for (int i = 0; i < numBodies; i++)
    {
    	accl += FORCE_CONSTANT * particle_particle(pos, curPos[i]);
    }

    return accl;
}

//Calculates the total position sum of all particles (each handled by one thread) in a block
//The fouth component is the number of particles in its neighbor radius
__device__ float4 compute_sumPos(float4 currentpos, float4* curPos, int numBodies)
{
    float3 pos = make_float3(0., 0., 0.);

    int j = 0;
    for (int i = 0; i < numBodies; i++)
    {
    	if (length(currentpos - curPos[i]) > FLOCKING_DIST)
    	{
    		continue;
    	}
    	j++;
		pos += VEC3(curPos[i]);
    }

    return make_float4(VEC3(pos), j);
}

//Calculates the total net acceleration on a particle from all other particles, based on both separation and cohesion flocking calculations
//Uses that acceleration to calculate velocity, which is then used to updated position in the global array 
__global__ void interact_kernel(float4* newPos, float4* oldPos, float4* newVel, float4* oldVel, float dt, float damping, int numBodies)
{
    // :: update positions based on particle-particle interaction forces!

	extern __shared__ float4 blockPositions[];

	unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	float4 currentPosition = oldPos[globalId];

	float3 accl = make_float3(0., 0., 0.);
	float3 sumpos = make_float3(0., 0., 0.);
	int inNeighbor = 0;

	for (int i = 0; i < gridDim.x; i++)
	{
		blockPositions[threadIdx.x] = oldPos[WRAP(i+blockIdx.x, gridDim.x) * blockDim.x + threadIdx.x];
		__syncthreads();

		accl += compute_accel(currentPosition, blockPositions, blockDim.x);
		float4 currentpos = compute_sumPos(currentPosition, blockPositions, blockDim.x);
		sumpos += VEC3(currentpos);
		inNeighbor+=(int)currentpos.w;

		__syncthreads();
	}

	//We treat the average position as a particle that we are gravitationally attracted to
	float3 avgpos = sumpos / inNeighbor;
	accl -= FLOCK_CONSTANT * particle_particle(currentPosition, make_float4(avgpos, 1.));

	//Using Symplectic Euler Integration: set position and velocity in global memory
	float3 vel = VEC3(oldVel[globalId]) * damping;
	vel += accl * dt;
	vel = VELOCITY_CONSTANT * UNIT(vel);
	newVel[globalId] = make_float4(vel, 1.);

	float3 pos = VEC3(oldPos[globalId]);
	pos += vel * dt;
	newPos[globalId] = make_float4(pos, 1.);

}	



