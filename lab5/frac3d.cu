/*
* Lab 5 - Fractal Volume Rendering
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_gl_interop.h>
#include <cutil_inline.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define REACHED printf("REACHED\n"); fflush(stdout);


#define CUBE_DIM 256

#include <frac3d_kernel.cu>

cudaExtent volumeSize = make_cudaExtent(CUBE_DIM,CUBE_DIM,CUBE_DIM);
size_t totalSize = 0;

/* Screen drawing kernel parameters */
uint width = 512, height = 512;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

/* Recompute kernel parameters */
dim3 volBlockSize(volumeSize.width);
dim3 volGridSize(volumeSize.height, volumeSize.depth);

/* View settings */
float3 viewRotation = make_float3(0.5, 0.5, 0.0);
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

/* Local Julia Set parameters */
float4 juliaC;
float4 juliaPlane;

/* Volume rendering isosurface */
float epsilon = 0.0025f;

/* Recalculate next frame? */
bool recompute = true;

// Pointer to "volume texture" cudaArray
cudaArray *d_volumeArray = 0;
// Pointer to global memory array
float *d_volumeData = NULL;
// OpenGL buffer pixel buffer
GLuint pbo = 0;
// Pitched pointer and params
cudaPitchedPtr d_volumePtr = {0};
cudaMemcpy3DParms d_volumeParams = {0};

void initPixelBuffer();

/* Execute the recompute kernel */
void recalculate()
{

    // begin timing code
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // run kernel

    // note: want to write to d_volumeData 
    /* : call d_setfractal kernel */

    d_setfractal<<<volGridSize, volBlockSize>>>( d_volumeData );
    cutilCheckMsg("kernel failedfwfaf");

    // copy data to volume texture
    /* : copy global memory to texture array */
    cudaMemcpy3D(&d_volumeParams);

    // end timing code
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Recompute took %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* Execute volume rendering kernel */
void render()
{
    // set necessary constants in hardware
    cutilSafeCall( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeof(float4)*3) );

    // recalculate set if we need to
    if (recompute)
    {
    	recalculate();
    	recompute = false;
    }

    // map PBO to get CUDA device pointer
    uint *d_output;
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_output, pbo));
    cutilSafeCall(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    /* : execute volume rendering kernel */

    d_render <<< gridSize, blockSize >>> (d_output, width, height, epsilon);
    cutilCheckMsg("kernel failed");

    cutilSafeCall(cudaGLUnmapBufferObject(pbo));
}

// display results using OpenGL (called by GLUT)
void display()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
        glLoadIdentity();
        glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
        glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
        glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    // transpose matrix to conform with OpenGL's notation
    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
}

//Set the Constants into Constant Memory
void setConsts()
{
    /* : Set c_juliaC and c_juliaPlane constants */
    cutilSafeCall(cudaMemcpyToSymbol(c_juliaC, &juliaC, sizeof(float4)));
    cutilSafeCall(cudaMemcpyToSymbol(c_juliaPlane, &juliaPlane, sizeof(float4)));
}

void keyboard(unsigned char key, int x, int y)
{
    float ds = 0.005f;

    switch(key) {
        case 27:
            exit(0);
            break;

        case 'w':
	    juliaC.y += ds;
	    recompute = true;
	    break;
        case 's':
	    juliaC.y -= ds;
	    recompute = true;
	    break;
        case 'a':
	    juliaC.x -= ds;
	    recompute = true;
	    break;
        case 'd':
	    juliaC.x += ds;
	    recompute = true;
	    break;
        case 'q':
	    juliaC.z -= ds;
	    recompute = true;
	    break;
        case 'e':
	    juliaC.z += ds;
	    recompute = true;
	    break;
        case 'z':
	    juliaC.w -= ds;
	    recompute = true;
	    break;
        case 'x':
	    juliaC.w += ds;
	    recompute = true;
	    break;

        case 'i':
	    juliaPlane.y += ds;
	    recompute = true;
	    break;
        case 'k':
	    juliaPlane.y -= ds;
	    recompute = true;
	    break;
        case 'j':
	    juliaPlane.x -= ds;
	    recompute = true;
	    break;
        case 'l':
	    juliaPlane.x += ds;
	    recompute = true;
	    break;
        case 'u':
	    juliaPlane.z -= ds;
	    recompute = true;
	    break;
        case 'o':
	    juliaPlane.z += ds;
	    recompute = true;
	    break;
        case 'm':
	    juliaPlane.w -= ds;
	    recompute = true;
	    break;
        case ',':
	    juliaPlane.w += ds;
	    recompute = true;
	    break;
	    

        case '=':
        case '+':
            epsilon *= 1.2;
	    printf("epsilon = %.5f\n", epsilon);
            break;
        case '-':
            epsilon /= 1.2;
	    printf("epsilon = %.5f\n", epsilon);
            break;

        default:
            break;
    }

    setConsts();

    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    width = x; height = y;
    // reinitialize with new size
    initPixelBuffer();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void initCuda()
{
    // create 3d cudaArray (in texture memory)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cutilSafeCall( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

    // create writeable 3d data array (in global memory)
    cutilSafeCall( cudaMalloc((void**)&d_volumeData, totalSize));
    d_volumePtr = make_cudaPitchedPtr((void*)d_volumeData, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);

    // set up copy params for future copying
    d_volumeParams.srcPtr   = make_cudaPitchedPtr((void*)d_volumeData, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    d_volumeParams.dstArray = d_volumeArray;
    d_volumeParams.extent   = volumeSize;
    d_volumeParams.kind     = cudaMemcpyDeviceToDevice;

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    cutilSafeCall(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

void cleanup()
{
    cutilSafeCall(cudaFree(d_volumeData));
    cutilSafeCall(cudaFreeArray(d_volumeArray));
    cutilSafeCall(cudaGLUnregisterBufferObject(pbo));    
    glDeleteBuffersARB(1, &pbo);
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer()
{
    if (pbo) {
        // delete old buffer
        cutilSafeCall(cudaGLUnregisterBufferObject(pbo));
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    
    cutilSafeCall(cudaGLRegisterBufferObject(pbo));

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
    
    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice( &device );
    cudaGetDeviceProperties( &prop, device );
    if( !strncmp( "Tesla", prop.name, 5 ) ){
        printf("This sample needs a card capable of OpenGL and display.\n");
        printf("Please choose a different device with the -device=x argument.\n");
        cutilExit(argc, argv);
    }

    // parse arguments
    int n;
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "size", &n)) {
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "xsize", &n)) {
        volumeSize.width = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "ysize", &n)) {
        volumeSize.height = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "zsize", &n)) {
         volumeSize.depth = n;
    }

    totalSize = sizeof(float)*volumeSize.width*volumeSize.height*volumeSize.depth;
    initCuda();

    printf("Press '=' and '-' to change epsilon\n"
           "      'qe,ws,ad,zx' to change c'\n"
           "      'jl,ik,m<' to change plane\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width*2, height*2);
    glutCreateWindow("CUDA volume rendering");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
    initPixelBuffer();

    /* Give c, plane initial values */
    juliaC = make_float4(-0.08f,0,-0.83f,-0.035f);
    juliaPlane = make_float4(0.3f,0.2f,-0.2f,0);
    setConsts();

    atexit(cleanup);

    glutMainLoop();

    cudaThreadExit();
    return 0;
}
