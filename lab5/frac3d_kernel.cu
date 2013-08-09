/*
 * Lab 5 - Fractal Volume Rendering
 */

#ifndef _TEXTURE3D_KERNEL_H_
#define _TEXTURE3D_KERNEL_H_

#include "cutil_math.h"

/* Volume texture declaration */
texture<float, 3, cudaReadModeElementType> tex;

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

/* Need to write host code to set these */
__constant__ float4 c_juliaC; // julia set constant
__constant__ float4 c_juliaPlane; // plane eqn of 3D slice


struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

#define ambientFactor .5f
#define diffuseFactor .5f
#define epsDistanceFactor 1.0f
#define tStepJuliaDistFactor .3f


// multiply two quaternions
__device__ float4
mul_quat(float4 p, float4 q)
{
    return make_float4(p.x*q.x-p.y*q.y-p.z*q.z-p.w*q.w,
		       p.x*q.y+p.y*q.x+p.z*q.w-p.w*q.z,
		       p.x*q.z-p.y*q.w+p.z*q.x+p.w*q.y,
		       p.x*q.w+p.y*q.z-p.z*q.y+p.w*q.x);
}

// square a quaternion (could be optimized)
__device__ float4
sqr_quat(float4 p)
{
    // this could/should be optimized
    return mul_quat(p,p);
}

// convert a 3d position to a 4d quaternion using plane-slice
__device__ float4
pos_to_quat(float3 pos, float4 plane)
{
    return make_float4(pos.x, pos.y, pos.z,
		       plane.x*pos.x+plane.y*pos.y+plane.z*pos.z+plane.w);
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

// color conversion functions
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ uint rFloatToInt(float r)
{
    r = __saturatef(r);   // clamp to [0.0, 1.0]
    return (uint(r*255)<<24) | (uint(r*255)<<16) | (uint(r*255)<<8) | uint(r*255);
}

// get a normal from volume texture
/* feel free to use this, but you should also compute the normal
   using JuliaDist */
__device__ float3 d_TexNormal(float3 pos)
{
    float3 normal = make_float3(0);
    float d = 0.04f;

    normal.x = (tex3D(tex, pos.x+d, pos.y, pos.z)-tex3D(tex, pos.x-d, pos.y, pos.z));
    normal.y = (tex3D(tex, pos.x, pos.y+d, pos.z)-tex3D(tex, pos.x, pos.y-d, pos.z));
    normal.z = (tex3D(tex, pos.x, pos.y, pos.z+d)-tex3D(tex, pos.x, pos.y, pos.z-d));

    return normalize(normal);
}


// computes julia distance function
__device__ float
d_JuliaDist(float3 pos, int niter)
{
    /* : fill in JuliaDist function */

    float4 z = pos_to_quat(pos, c_juliaPlane);
    float4 zp = make_float4(1.f, 0.f, 0.f, 0.f);
    float norm;
    float normsquared;

    int counter = 0;
    while (1)
    {
        //norm = length(sqr_quat(z, z));
        norm = length(z);
        normsquared = norm * norm;
        if (normsquared > 20.f || counter > niter)
        {
            break;
        }
        
        zp = 2.f * mul_quat(z, zp); 
        z = mul_quat(z, z) + c_juliaC;
        counter++;
    }

    float zlength = length(z);
    float zplength = length(zp);
    return zlength / (2.f * zplength) * log(zlength);
}


/// compute a normal using juliaDist
__device__ float3 d_JuliaNormal(float3 pos, int niter)
{
    float dt = 1.f/512.f;

    float E     = d_JuliaDist (make_float3(pos.x - dt, pos.y, pos.z), niter);
    float W     = d_JuliaDist (make_float3(pos.x + dt, pos.y, pos.z), niter);
    float N     = d_JuliaDist (make_float3(pos.x, pos.y + dt, pos.z), niter);
    float S     = d_JuliaDist (make_float3(pos.x, pos.y - dt, pos.z), niter);
    float Near  = d_JuliaDist (make_float3(pos.x, pos.y, pos.z - dt), niter);
    float Far   = d_JuliaDist (make_float3(pos.x, pos.y, pos.z + dt), niter);

    float3 norm =  normalize  (make_float3( E - W, N - S, Far - Near ) / (2.*dt));
    return norm;
}

// perform volume rendering
__global__ void
d_render(uint *d_output, uint imageW, uint imageH, float epsilon)
{

    float3 lightPos = make_float3(5.0, 3.0, 5.0);
    // amount to step by
    float tstep = 0.003f;
    int maxSteps = 2000;

    //iterations needed for JuliaDist call
    int niter = 50;

    
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;

    // return if not intersecting
    if (!intersectBox(eyeRay, 
		      make_float3(-2.0f,-2.0f,-2.0f), 
		      make_float3(2.0f,2.0f,2.0f),
		      &tnear, &tfar))
	return;

    // clamp to near plane
    if (tnear < 0.0f) tnear = 0.0f;

    float t = tnear;

    // accumulate values
    float accum = 0.0f;
    float3 pos;
    
    // start stepping through space
    for(int i=0; i<maxSteps; i++) {
        float eps = epsilon;		
        pos = eyeRay.o + eyeRay.d*t;
        float distFromCamera = length(pos - eyeRay.o);
        eps *= epsDistanceFactor * distFromCamera;

        tstep = tStepJuliaDistFactor * d_JuliaDist(pos, niter);
        
        // map position to [0, 1] coordinates
        pos = pos*0.25f+0.5f;

        // read from 3D texture
        float sample = tex3D(tex, pos.x, pos.y, pos.z);

	    accum += sample;

        t += tstep;

        if (sample < eps) break;
        if (t > tfar) return;
    }

    /* : calculate normal vector */
    float3 texpos (4.f * (pos - 0.5f));
    float3 N = d_JuliaNormal(texpos, niter);

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;

	/* : calculate output color based on lighting and position */
        float4 posColor = make_float4(pos, 0.);
        float4 col4 = make_float4(0., 0., 0., 1.);
        float4 ambient = ambientFactor * (posColor);

        float3 L = normalize(lightPos - eyeRay.o);
        float NdotL = max(dot(N, L), 0.0);
        float4 diffuse = diffuseFactor * posColor * NdotL;

        col4 += (ambient + diffuse);

        d_output[i] = rgbaFloatToInt(col4);
    }
}

// recompute julia set at a single volume point
__global__ void
d_setfractal(float *d_output)
{
    int niter = 100;

    //CURRENTLY COALESCED
    // get x,y,z indices from kernel
    uint x = threadIdx.x;

    /* : get y, z coordinates from blockIdx,
       compute juliadist at corresponding (normalized) position */
    uint y = blockIdx.x;
    uint z = blockIdx.y;
    uint width = blockDim.x;
    uint height = gridDim.x;

    float3 pos = make_float3( 4.0f * ((float) x/(float) blockDim.x - 0.5f), 4.0f * ((float) y/(float) gridDim.x - 0.5f), 4.0f * ((float) z/(float) gridDim.y - 0.5f) );
    ulong i = x + width*y + width*height*z;

    // set output value
    float dist = d_JuliaDist(pos, niter);
    d_output[i] = dist;
}

#endif // #ifndef _TEXTURE3D_KERNEL_H_
