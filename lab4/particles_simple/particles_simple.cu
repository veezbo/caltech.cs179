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


////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// vbo variables
GLuint vbo;

// Device buffer variables
float4* d_velocities;

float anim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

////////////////////////////////////////////////////////////////////////////////
// kernels
__global__ void simple_kernel( float4* pos, float4* vels, unsigned int width, unsigned int height, float time, float rand, float prand); 

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runParticles( int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createVBO( GLuint* vbo);
void deleteVBO( GLuint* vbo);


// rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda( GLuint vbo);
void checkResultCuda( int argc, char** argv, const GLuint& vbo);

void createDeviceData();
void deleteDeviceData();


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    runParticles( argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple particle simulation for CUDA
////////////////////////////////////////////////////////////////////////////////
void runParticles( int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "Simple Particles");

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
    createVBO( &vbo);

    // create other device memory (velocities!)
    createDeviceData();

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda( GLuint vbo)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, vbo));

    float dt = 0.01;

    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_kernel<<< grid, block>>>(dptr, d_velocities, 
                                        mesh_width, mesh_height, dt,
                                        ((float)rand())/RAND_MAX,
                                        ((float)rand())/RAND_MAX);

    // unmap buffer object
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo));
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
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // Initialize data.
    float4* temppos = (float4*)malloc(mesh_width*mesh_height*4*sizeof(float));
    for(int i = 0; i < mesh_width*mesh_height; ++i)
    {
        // temppos[i].x = ((float)rand())/RAND_MAX;
        // temppos[i].y = ((float)rand())/RAND_MAX;
        // temppos[i].z = ((float)rand())/RAND_MAX;
        temppos[i].x = 2.*((float)rand())/RAND_MAX-1.;
        temppos[i].y = 2.*((float)rand())/RAND_MAX-1.;
        temppos[i].z = 2.*((float)rand())/RAND_MAX-1.;
        temppos[i].w = 1.;
    }

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof( float);
    glBufferData( GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vbo));

    free(temppos);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vbo));

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create device data
////////////////////////////////////////////////////////////////////////////////
void createDeviceData()
{
    // Create a velocity for every position.
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_velocities, mesh_width*mesh_height*
                                                    4 * sizeof(float) ) );

    // Initialize data.
    float4* tempvels = (float4*)malloc(mesh_width*mesh_height*4*sizeof(float));
    for(int i = 0; i < mesh_width*mesh_height; ++i)
    {
        tempvels[i].x = 2.*((float)rand())/RAND_MAX-1.;
        tempvels[i].y = 2.*((float)rand())/RAND_MAX-1.;
        tempvels[i].z = 2.*((float)rand())/RAND_MAX-1.;
        tempvels[i].w = ((float)rand())/RAND_MAX;
    }

    // Copy to gpu
    CUDA_SAFE_CALL( cudaMemcpy( d_velocities, tempvels, mesh_width*mesh_height*4*sizeof(float), cudaMemcpyHostToDevice) );
    free(tempvels);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete device data
////////////////////////////////////////////////////////////////////////////////
void deleteDeviceData()
{
    // Create a velocity for every position.
    CUDA_SAFE_CALL( cudaFree( d_velocities ) );
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions
    runCuda(vbo);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

    anim += 0.01;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case( 27) :
        deleteVBO( &vbo);
        deleteDeviceData();
        exit( 0);

    case( 81) :
        deleteVBO( &vbo);
        deleteDeviceData();
        exit( 0);

    case( 113) :
        deleteVBO( &vbo);
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
