#include <iostream>

#define GL_GLEXT_PROTOTYPES

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include "Mesh.h"
#include "uistate.h"
#include "glFunctions.h"
#include <stdlib.h>
using namespace std;

// Peter's mystical ui controller for arcball transformation and stuff
static UIState *ui;

// The mesh being displayed
Mesh *mesh;

// A simple compiled shader program.
const char *simpleVertSource = "simple.vert", *simpleFragSource = "simple.frag";
GLuint simpleProgram;

// A texture that can be used in shaders or fixed function rendering.  Here
// used to make the background image.
GLuint backgroundTexture;

// The current window size.
int windowWidth = 800, windowHeight = 600;

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

int main(int argc, char *argv[]);

//--------------------------------------------------------------------------
// Calls the current display function.
//--------------------------------------------------------------------------
void display()
{
    checkGLErrors("Beginning of display");
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Render a nice DeviantArt background for fun.
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    // Don't use a program.  That is, use the fixed funtion pipeline.
    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, backgroundTexture);
    renderFullscreenQuad();
    // But make sure we get back to normal afterwards.
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    // Done with the goofy backround stuff.
    
    // Use our simple program.
    glUseProgram(simpleProgram);
    // Apply the camera transformation.
    ui->ApplyViewingTransformation();
    
    // Render the mesh.
    int numFaces = mesh->face.size();
    for (int i = 0; i < numFaces; i++)
    {
        Triangle &face = mesh->face[i];
        glBegin(GL_POLYGON);
        for (int j = 0; j < 3; j++)
        {
            Vertex *vert = face.vertex[j];
            glNormal3f(vert->normal.x(), vert->normal.y(), vert->normal.z());
            glVertex3f(vert->position.x(), vert->position.y(), vert->position.z());
        }
        glEnd();
    }
    
    glutSwapBuffers();
    checkGLErrors("End of display");
}

//--------------------------------------------------------------------------
// Handles keypresses.  Translates them into mode changes, etc.
//--------------------------------------------------------------------------
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27: // ESC
        case 'q':
        case 'Q':
            exit(0);
    }
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
    ui->Radius() = 2;
    ui->Near() = .1;
    ui->Far() = 10;
    ui->CTrans().z() = -2;
    reshape(windowWidth, windowHeight);
    checkGLErrors("End of uiInit");
}

//--------------------------------------------------------------------------
// Load things from files.
//--------------------------------------------------------------------------
void loadData()
{
    createTexture(backgroundTexture, "BillsBackground.png");
    
    readShader(simpleVertSource, simpleFragSource, simpleProgram);
    
    mesh = new Mesh();
    //if (!mesh->parseOFF("Data/fandisk.off"))
    if (!mesh->parseOBJ("Data/cow.obj"))
    {
        cerr << "Unable to load mesh.  Aborting." << endl;
        exit(0);
    }
}

//--------------------------------------------------------------------------
// Main entry point.
//--------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    
    glutInitWindowSize(windowWidth, windowHeight);
    
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("GPU Course Prototype");
    
    initGL();
    
    initUI();
    
    // Load all of our personalized data.
    loadData();
    
    glutDisplayFunc(display);
    glutIdleFunc(glutPostRedisplay);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    
    glutMainLoop();
    return 0;
}
