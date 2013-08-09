#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#define GL_GLEXT_PROTOTYPES

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glext.h>
#endif

using namespace std;

#include "uistate.h"
#include "glFunctions.h"
#include <stdlib.h>

// Peter's mystical ui controller for arcball transformation and stuff
static UIState *ui;

// The physics shader
char *physicsVertSource, *physicsFragSource;
GLuint physicsProgram;

char *energyFragSource;
GLuint energyProgram;

char *reductionFragSource;
GLuint reductionProgram;

// Textures in graphics memory
GLuint positionTex[2];
GLuint velocityTex[2];
GLuint energyTex[2];
GLuint spriteTex;

//texture handles for shaders
GLuint velTexLoc;
GLuint posTexLoc;
GLuint randLoc;

// Width and height for the particle textures.
const int w = 256; //make sure these values are the same as in reduction.frag
const int h = 256;

float averageEnergy = 0.0;

//frame buffer object
GLuint fbo;

//vertex buffer object
GLuint vbo;

// our MRT buffers
GLenum MRTBuffers[2][2] = {{ GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT }, 
                           { GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT }};

GLenum ENGBuffers[2] = {GL_COLOR_ATTACHMENT4_EXT, GL_COLOR_ATTACHMENT5_EXT};

// The current window size.
int windowWidth = 800, windowHeight = 600;

// which texture group are we writing to?
int texToWrite = 1;

// Calls the current display function
void display();
// Handles keypresses.  Translates them into mode changes, etc.
void keyboard(unsigned char key, int x, int y);
// Handles reshaping of the program window.
void reshape(const int width, const int height);
// Handles motion of the mouse when a button is being held.
void motion(const int x, const int y);
// Handles mouse clicks and releases.
void mouse(int button, int state, int x, int y);
// Initializes the UI.
void initUI();
// Load things from files.
void loadData();
// Handy function to ensure frame buffer is okay.
bool checkFramebufferStatus();


int main(int argc, char *argv[]);



//--------------------------------------------------------------------------
// Runs the current positionProgram on the current textures and framebuffer state such
// that a box from xMin,yMin to xMax,yMax is processed by the positionProgram.  Takes
// care of resizing the viewing window and providing a 1:1 texel:pixel mapping.
//--------------------------------------------------------------------------
void processPixels(int width, int height) {

    // Set up projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, width, 0, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);

    // Render full screen quad with tex coords
    glBegin(GL_POLYGON);
    glTexCoord2f(0, 0);
    glVertex3f(0, 0, 0);
    glTexCoord2f(0, height);
    glVertex3f(0, height, 0);
    glTexCoord2f(width, height);
    glVertex3f(width, height, 0);
    glTexCoord2f(width, 0);
    glVertex3f(width, 0, 0);
    glEnd();
    reshape(windowWidth, windowHeight);
}

unsigned int pp_Index = 0;

void processPhysics()
{
    //first, bind the fbo
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);  

    // Bind the right textures to read from.  Texture Unit 0 should be velocity, 1 should be position.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, velocityTex[pp_Index]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, positionTex[pp_Index]);

    // Use physics program, set random uniforms.
    glUseProgram(physicsProgram);
    randLoc = glGetUniformLocation(physicsProgram, "rand");
    glUniform1f(randLoc, ((float)rand())/RAND_MAX);
    randLoc = glGetUniformLocation(physicsProgram, "prand");
    glUniform1f(randLoc, ((float)rand())/RAND_MAX);

    // Set the correct pair of Draw Buffers.
    glDrawBuffers(2, MRTBuffers[(pp_Index + 1) % 2]);

    // Render to current draw buffers.
    checkFramebufferStatus();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    processPixels(w,h); 

    // Ping-pong textures
    pp_Index = (pp_Index + 1) % 2;
}

void renderParticles()
{
    // Render the particles from the VBO

    // Load the VBO with data
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

    // Set the read buffer
    glReadBuffer(MRTBuffers[pp_Index][1]);

    // We're going to read pixels into the vbo
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB,vbo);

    checkFramebufferStatus();

    // Perform the read operation, starting at offset 0 from the vbo
    glReadPixels(0,0,w,h,GL_RGB,GL_FLOAT, 0);

    checkFramebufferStatus();

    // Bind and enable the VBO as the vertex array buffer
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glVertexPointer(3,GL_FLOAT, 0, NULL);


    // Now that we've loaded and bound the VBO, we can render the data as points.

    // Unbind framebuffer - we're rendering to screen
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glColor3f(1,0,0);

    // Use fixed-functionality
    glUseProgram(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glTranslated(-.5,-.5,-.5);
    glPointSize(2);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, w*h);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void calculateEnergy()
{
    // Calculate average particle energy.

    // Bind the frame buffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);  

    // Set the read textures.  Again, texture 0 should be velocity, 1 should be position.  
    // Be mindful that this is called after the processPhysics function every frame.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, velocityTex[pp_Index]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, positionTex[pp_Index]);

    // Use energy program
    glUseProgram(energyProgram);

    // Process the energy into 4th color attachment.
    glDrawBuffer(GL_COLOR_ATTACHMENT4_EXT);
    checkFramebufferStatus();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    processPixels(w,h);

    // Use the reduction shader to repeatedly half the dimensions of the energy texture until it is 1x1.
    // Ping-pong the read and write textures as you go.  
    int i = 0;
    for(int dim = w; dim > 1; i++, dim /= 2)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, energyTex[i % 2]);
        glUseProgram(reductionProgram);
        glDrawBuffer(ENGBuffers[(i + 1)%2]); //defined this up top
                                             //contains colorattach 4/5
        checkFramebufferStatus();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        processPixels(dim, dim);
    }
    
    // Bind the correct read buffer for the ReadPixels.
    glReadBuffer(ENGBuffers[i % 2]); // i would be incremented so just i%2

    // Unbind the pixel pack buffer so we can read to CPU memory.
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB,0);

    // Read in the rgba value, and set the average energy to the r value.
    checkGLErrors("Pre-Read");
    float rgba[4];
    glReadPixels(0,0,1,1,GL_RGBA, GL_FLOAT, &rgba);
    averageEnergy = rgba[0];
    checkGLErrors("End of calculate energy");
}

// Gets called at startup.  Put physics set up here.
void initPhysics()
{
    float posPixels[w*h][4];
    float velPixels[w*h][3];
    srand(time(0));
   
    // uniform height and vertical velocity
    float y_rand = ((float) rand())/RAND_MAX;
    float vy_rand = ((float) rand())/RAND_MAX;

    for(int i = 0; i < w*h; i++)
    {
        // : give your pixels some initial data
        velPixels[i][0] = 0.;
        velPixels[i][1] = 0.;
        velPixels[i][2] = 0.;
        float theta = 2.*3.14*((float)rand())/RAND_MAX;
        posPixels[i][0] = cosf(theta) + 0.5;
        posPixels[i][1] = sinf(theta) + 0.5;
        posPixels[i][2] = 0.05*((float)rand())/RAND_MAX;
        posPixels[i][3] = ((float)rand())/RAND_MAX;
    }

    createTexture(velocityTex[0], velPixels, w, h, GL_RGBA32F_ARB, GL_FLOAT, GL_RGB); 
    createTexture(positionTex[0], posPixels, w, h, GL_RGBA32F_ARB, GL_FLOAT, GL_RGBA); 
    createTexture(velocityTex[1], velPixels, w, h, GL_RGBA32F_ARB, GL_FLOAT, GL_RGB); 
    createTexture(positionTex[1], posPixels, w, h, GL_RGBA32F_ARB, GL_FLOAT, GL_RGBA); 

    // Since we write to the energy textures before we read them, we can fill them with junk.
    createTexture(energyTex[0], posPixels, w, h, GL_RGBA32F_ARB, GL_FLOAT, GL_RGBA);
    createTexture(energyTex[1], posPixels, w, h, GL_RGBA32F_ARB, GL_FLOAT, GL_RGBA);
}

void initBuffers()
{
    //Create Frame and Vertex Buffer.
    
    //create VBO
    glGenBuffersARB(1, &vbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, w*h*3*sizeof(float), 0, GL_STREAM_DRAW_ARB);

    //set-up fbo
    glGenFramebuffersEXT(1, &fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, velocityTex[0], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, positionTex[0], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_RECTANGLE_ARB, velocityTex[1], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT3_EXT, GL_TEXTURE_RECTANGLE_ARB, positionTex[1], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT4_EXT, GL_TEXTURE_RECTANGLE_ARB, energyTex[0], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT5_EXT, GL_TEXTURE_RECTANGLE_ARB, energyTex[1], 0);
    //Allocate energy buffer and bind it to our framebuffer.
    unsigned int energybuffer;
    glGenRenderbuffersEXT(1, &energybuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, energybuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, energybuffer);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

}

bool checkFramebufferStatus() {
    GLenum status;
    status = (GLenum) glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            return true;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
            printf("Framebuffer incomplete, incomplete attachment\n");
            return false;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            printf("Unsupported framebuffer format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            printf("Framebuffer incomplete, missing attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            printf("Framebuffer incomplete,  attachments must have same dimensions\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
            printf("Framebuffer incomplete, attached images must have same format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            printf("Framebuffer incomplete, missing draw buffer\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            printf("Framebuffer incomplete, missing read buffer\n");
            return false;
    }
    return false;
}

void printEnergy()
{
    char energyBuffer[100];
    snprintf(energyBuffer, 100, "Average Absolute Energy: %f", averageEnergy);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, windowWidth, 0, windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, windowWidth, windowHeight);

    glColor3f(0,1,0);
    for(int i = 0; i < strlen(energyBuffer); i++)
    {
        glRasterPos3f(500 + 9*i, windowHeight - 20, 0);
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, energyBuffer[i]);
    }
}

void display()
{
    // update animation variable
//    animTime += 0.0005f;

    checkGLErrors("Beginning of display");
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Process changes in velocity and position.
    processPhysics();

    // Calculate Energy
    calculateEnergy();

    // Apply the camera transformation.
    ui->ApplyViewingTransformation();

    // render the particles
    renderParticles();

    // print the energy string
    printEnergy();

    glutSwapBuffers();
    checkGLErrors("End of display");
}

//--------------------------------------------------------------------------
// Handles keypresses.  Translates them into mode changes, etc.
//--------------------------------------------------------------------------
void keyboard(unsigned char key, int x, int y)
{
    if(key == 27) // ESC
        exit(0);
    if(key == 'q' || key == 'Q')
        exit(0);
}

//--------------------------------------------------------------------------
// Handles reshaping of the program window
//--------------------------------------------------------------------------
void reshape(const int width, const int height)
{
    windowWidth = width;
    windowHeight = height;
    
    if( width <= 0 || height <= 0 ) return;
    
    ui->WindowX() = width;
    ui->WindowY() = height;
    
    ui->Aspect() = float( width ) / height;
    ui->SetupViewport();
    ui->SetupViewingFrustum();
}

//--------------------------------------------------------------------------
// Handles motion of the mouse when a button is being held
//--------------------------------------------------------------------------
void motion(const int x, const int y)
{
    // Just pass it on to the ui controller.
    ui->MotionFunction(x, y);
}

//--------------------------------------------------------------------------
// Handles mouse clicks and releases
//--------------------------------------------------------------------------
void mouse(const int button, const int state, const int x, const int y)
{
    // Just pass it on to the ui controller.
    ui->MouseFunction(button, state, x, y);
}

//--------------------------------------------------------------------------
// Initializes the UI
//--------------------------------------------------------------------------
void initUI()
{
    ui = new UIState;
    ui->Trans() = Vector3(0, 0, 0);
    ui->Radius() = 1;
    ui->Near() = .1;
    ui->Far() = 10;
    ui->CTrans().z() = -1;
    reshape(windowWidth, windowHeight);
    checkGLErrors("End of uiInit");
}

//--------------------------------------------------------------------------
// Load things from files.
//--------------------------------------------------------------------------
void loadFromRAW(const char* filename, char *data, int size)
{
    FILE* f = fopen(filename, "r");
    if (f == NULL)
	fprintf(stderr,"file not found!\n");

    int n = fread(data, size, 1, f);
    if (n != 1)
	fprintf(stderr, "Error reading file!\n");

    fclose(f);
} 

void loadData()
{
    readShader((const char*)physicsVertSource, (const char*)physicsFragSource, physicsProgram);
    velTexLoc = glGetUniformLocation(physicsProgram, "velocityTex");
    posTexLoc = glGetUniformLocation(physicsProgram, "positionTex");
    glUniform1i(velTexLoc, 0);
    glUniform1i(posTexLoc, 1);

    readShader((const char*)physicsVertSource, (const char*)energyFragSource, energyProgram);
    velTexLoc = glGetUniformLocation(energyProgram, "velocityTex");
    posTexLoc = glGetUniformLocation(energyProgram, "positionTex");
    glUniform1i(velTexLoc, 0);
    glUniform1i(posTexLoc, 1);

    readShader((const char*)physicsVertSource, (const char*)reductionFragSource, reductionProgram);
    velTexLoc = glGetUniformLocation(reductionProgram, "reductionTex");
    glUniform1i(velTexLoc, 0);

    glUseProgram(0);

}

//--------------------------------------------------------------------------
// Main entry point.
//--------------------------------------------------------------------------
int main(int argc, char **argv)
{
    glutInit(&argc, argv);

    physicsVertSource = "physics.vert";
    physicsFragSource = "physics.frag";
    energyFragSource = "energy.frag";
    reductionFragSource = "reduction.frag";
    
    glutInitWindowSize(windowWidth, windowHeight);
    
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("CS101 Lab 4");

    initGL();
    
    initUI();
    
    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glDisable(GL_LIGHTING);
    // Load all of our personalized data.
    loadData();

    initPhysics();

    initBuffers();

    glutDisplayFunc(display);
    glutIdleFunc(glutPostRedisplay);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    
    glutMainLoop();
    return 0;
}
