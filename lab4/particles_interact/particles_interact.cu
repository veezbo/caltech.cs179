
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes
#include <cutil.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <assert.h>

//print include- for debugging purposes
//#include "cuPrintf.cuh"


////////////////////////////////////////////////////////////////////////////////
// constants
#define BLOCK_SIZE 128 // :: Choose one!       // Number of threads in a block.


unsigned int numBodies;     // Number particles; determined at runtime.

// window
int window_width = 512;
int window_height = 512;

// Flag for pingpong;
int pingpong = 0;

// vbo variables
GLuint vbo_pos[2];

// Device buffer variables
float4* dVels[2];


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;

////////////////////////////////////////////////////////////////////////////////
// kernels
__global__ void interact_kernel( float4* newPos, float4* oldPos, float4* newVel, float4* oldVel, float dt, float damping, int numBodies);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runParticles( int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createVBOs( GLuint* vbo);
void deleteVBOs( GLuint* vbo);


// rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda( GLuint *vbo, float dt);

void createDeviceData();
void deleteDeviceData();


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Usage: ./particles_interact N\n");
        exit(1);
    }

    numBodies = atoi(argv[1]);

    //cudaPrintfInit(numBodies * sizeof(float));
    runParticles( argc, argv);
    //cudaPrintfEnd();

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple particle simulation for CUDA
////////////////////////////////////////////////////////////////////////////////
void runParticles( int argc, char** argv)
{
    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( " Interacting Particles ");

    // Init random number generator
    srand(time(0));

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);

    // create VBO
    createVBOs( vbo_pos );

    // create other device memory (velocities!)
    createDeviceData();

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda( GLuint *vbo, float dt)
{
    // map OpenGL buffer object for writing from CUDA
    float4* oldPos;
    float4* newPos;

    // Velocity damping factor
    float damping = 0.995;

    // :: Map opengl buffers to CUDA.
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldPos, vbo[pingpong]));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&newPos, vbo[1-pingpong]));

    // :: Choose a block size, a grid size, an amount of shared mem,
    // and execute the kernel
    // dVels is the particle velocities old, new. Pingponging of these is
    // handled, if the initial conditions have initial velocities in dVels[0].

    dim3 block (BLOCK_SIZE, 1, 1);
    dim3 grid (numBodies / BLOCK_SIZE, 1, 1);
    unsigned int sharedMemSize = 4 * BLOCK_SIZE * sizeof(float);

    interact_kernel<<< grid, block, sharedMemSize >>>( newPos, oldPos, dVels[1-pingpong], dVels[pingpong], dt, damping, numBodies );
    //cudaPrintfDisplay(stdout, true);

    // :: unmap buffer objects from cuda.
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo[pingpong]));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo[1-pingpong]));

    // :: Switch buffers between old/new
    pingpong ^= 1;

}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    
    // :: depending on your parameters, you may need to change
    // near and far view distances (1, 500), to better see the simulation.
    // If you do this, probably also change translate_z initial value at top.
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 1, 500.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBOs(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 2, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo[0]);

    // initialize buffer object; this will be used as 'oldPos' initially
    unsigned int size = numBodies * 4 * sizeof( float);

    // :: Modify initial positions!
    float4* temppos = (float4*)malloc(numBodies*4*sizeof(float));
    for(int i = 0; i < numBodies; i++)
    {
        //Placing particles in each of the 4 2D quadrants
        int quadrant = floor(((float)rand())/RAND_MAX * 4.) + 1;

        if (quadrant == 1)
        {
            temppos[i].x = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].y = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].z = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].w = 1.;
        }
        else if (quadrant == 2)
        {
            temppos[i].x = -6.*((float)rand())/RAND_MAX - 1.5;
            temppos[i].y = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].z = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].w = 1.;    
        }
        else if (quadrant == 3)
        {
            temppos[i].x = -6.*((float)rand())/RAND_MAX - 1.5;
            temppos[i].y = -6.*((float)rand())/RAND_MAX - 1.5;
            temppos[i].z = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].w = 1.;    
        }
        else if (quadrant == 4)
        {
            temppos[i].x = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].y = -6.*((float)rand())/RAND_MAX - 1.5;
            temppos[i].z = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].w = 1.;    
        }
        else
        {
            temppos[i].x = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].y = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].z = 6.*((float)rand())/RAND_MAX + 1.5;
            temppos[i].w = 1.;
        }

    }

    // Notice only vbo[0] has initial data!
    glBufferData( GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    free(temppos);

    // Create initial 'newPos' buffer
    glBindBuffer( GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);


    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer objects with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo[0]));
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo[1]));

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBOs( GLuint* vbo)
{
    glBindBuffer( 1, vbo[0]);
    glDeleteBuffers( 1, &vbo[0]);
    glBindBuffer( 1, vbo[1]);
    glDeleteBuffers( 1, &vbo[1]);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo[0]));
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo[1]));

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create device data
////////////////////////////////////////////////////////////////////////////////
void createDeviceData()
{
    // :: Modify velocities if need be.
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dVels[0], numBodies *
                                                    4 * sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dVels[1], numBodies *
                                                    4 * sizeof(float) ) );

    // Initialize data.
    float4* tempvels = (float4*)malloc(numBodies * 4*sizeof(float));
    for(int i = 0; i < numBodies; ++i)
    {
        tempvels[i].x = .25*((2*(float)rand())/RAND_MAX-1);
        tempvels[i].y = .25*((2*(float)rand())/RAND_MAX-1);
        tempvels[i].z = .25*((2*(float)rand())/RAND_MAX-1);
        tempvels[i].w = 0.;
    }

    // Copy to gpu
    CUDA_SAFE_CALL( cudaMemcpy( dVels[0], tempvels, numBodies*4*sizeof(float), cudaMemcpyHostToDevice) );

    free(tempvels);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete device data
////////////////////////////////////////////////////////////////////////////////
void deleteDeviceData()
{
    // Create a velocity for every position.
    CUDA_SAFE_CALL( cudaFree( dVels[0] ) );
    CUDA_SAFE_CALL( cudaFree( dVels[1] ) );
    // pos's are the VBOs
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    runCuda(vbo_pos, 0.001);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo with newPos
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos[pingpong]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0, 1.0, 0.0);
    glDrawArrays(GL_POINTS, 0, numBodies);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case( 27) :
        deleteVBOs( vbo_pos );
        deleteDeviceData();
        exit( 0);

    case( 81) :
        deleteVBOs( vbo_pos );
        deleteDeviceData();
        exit( 0);

    case( 113) :
        deleteVBOs( vbo_pos );
        deleteDeviceData();
        exit( 0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
