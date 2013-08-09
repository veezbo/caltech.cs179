#include <iostream>
#include <fstream>

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

#include "readpng.h"
#include "glFunctions.h"
#include <stdlib.h>
using namespace std;

//--------------------------------------------------------------------------
// Sets up initial status of OpenGL.
//--------------------------------------------------------------------------
void initGL()
{
    glClearColor(0.75, 0.75, 0.75, 0);
    
    initLights();
    initMaterial();
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    
    checkGLErrors("End of initGL");
}

//--------------------------------------------------------------------------
// Sets up an OpenGL light.  This only needs to be called once
// and the light will be used during all renders.
//--------------------------------------------------------------------------
void initLights()
{
    GLfloat amb[]= { 0.0f, 0.0f, 0.0f, 1.0f };
    GLfloat diff[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat spec[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat lightpos[]= { 2.0f, 2.0f, 5.0f, 1.0f };
    
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diff);
    glLightfv(GL_LIGHT0, GL_SPECULAR, spec);
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
    glEnable(GL_LIGHT0);
    
    // Turn on lighting.  You can turn it off with a similar call to
    // glDisable().
    glEnable(GL_LIGHTING);
}

//--------------------------------------------------------------------------
// Sets the OpenGL material state.  This is remembered so we only need to
// do this once.  If you want to use different materials, you'd need to do this
// before every different one you wanted to use.
//--------------------------------------------------------------------------
void initMaterial()
{
    GLfloat emit[] = {0.0, 0.0, 0.0, 1.0};
    GLfloat  amb[] = {0.0, 0.0, 0.0, 1.0};
    GLfloat diff[] = {0.1, 0.7, 0.5, 1.0};
    GLfloat spec[] = {0.75, 1.0, 0.75, 1.0};
    GLfloat shiny = 20.0f;
    
    glMaterialfv(GL_FRONT, GL_AMBIENT, amb);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diff);
    glMaterialfv(GL_FRONT, GL_SPECULAR, spec);
    glMaterialfv(GL_FRONT, GL_EMISSION, emit);
    glMaterialfv(GL_FRONT, GL_SHININESS, &shiny);
}

//--------------------------------------------------------------------------
// Check OpenGL error status, and if an error is encountered, prints the error
// and label.
//--------------------------------------------------------------------------
void checkGLErrors(const char *label)
{
    GLenum errCode;
    const GLubyte *errStr;
    if ((errCode = glGetError()) != GL_NO_ERROR)
    {
        errStr = gluErrorString(errCode);
        cout << "OpenGL ERROR: " << (char*)errStr << "Label: " << label << endl;
    }
}

//--------------------------------------------------------------------------
// Initializes the texture with the supplied data as starting values.
//--------------------------------------------------------------------------
void createTexture(GLuint &texNum, GLvoid *pixels, int width, int height,
                   GLenum internalFormat, GLenum uploadType,
                   GLenum uploadFormat)
{
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &texNum);
    glBindTexture(GL_TEXTURE_2D, texNum);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat,
                 width, height,
                 0, uploadFormat, uploadType, pixels);
    // There are different options for these values, but if you don't set
    // them to something, the texture probably won't work at all.  Go figure.
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
                    GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    checkGLErrors("createTexture (raw data)");
}
//--------------------------------------------------------------------------
// Initializes a texture from a .png file.
//--------------------------------------------------------------------------
void createTexture(GLuint &texNum, const char *pngFilename,
                   GLenum internalFormat)
{
    int width, height;
    cout << "Attempting to read texture: " << pngFilename << endl;
    png_bytepp texData = readpng(pngFilename, &width, &height);
    if (texData == NULL)
    {
        cerr << "Failed to read texture." << endl;
        return;
    }
    char *tmp = (char*)malloc(width*height*3);
    for(int y=0; y < height; y++)
    {
        memcpy(tmp+width*y*3, texData[height-y-1], width*3);
    }
    createTexture(texNum, tmp, width, height, internalFormat,
                  GL_UNSIGNED_BYTE, GL_RGB);
    cout << "Texture load finished." << endl;
    free(texData);
    free(tmp);
    checkGLErrors("createTexture (png file)");
}

//--------------------------------------------------------------------------
// Prints an info log regarding the creation of a vertex or fragment shader
//--------------------------------------------------------------------------
void printShaderInfoLog(GLuint obj)
{
    GLint infologLength = 0;
    GLint charsWritten  = 0;
    char *infoLog;
    
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
    
    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf("%s\n",infoLog);
        free(infoLog);
    }
}

//--------------------------------------------------------------------------
// Prints an info log regarding the creation of an entire GPU program
//--------------------------------------------------------------------------
void printProgramInfoLog(GLuint obj)
{
    GLint infologLength = 0;
    GLint charsWritten  = 0;
    char *infoLog;
    
    glGetProgramiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
    
    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf("%s\n",infoLog);
        free(infoLog);
    }
}

//--------------------------------------------------------------------------
// Reads the shader at the supplied filenames, and compiles it into program.
//--------------------------------------------------------------------------
void readShader(const string vertProgFilename,
                const string fragProgFilename,
                GLuint &program)
{
    string vertProgramSource, fragProgramSource;
    ifstream vertProgFile(vertProgFilename.c_str());
    if (!vertProgFile)
    {
        cerr << "Error opening vertex shader program file." << endl;
        return;
    }
    ifstream fragProgFile(fragProgFilename.c_str());
    if (!fragProgFile)
    {
        cerr << "Error opening fragment shader program file." << endl;
        return;
    }
    
    getline(vertProgFile, vertProgramSource, '\0');
    const char *vertShaderSource = vertProgramSource.c_str();
    
    getline(fragProgFile, fragProgramSource, '\0');
    const char *fragShaderSource = fragProgramSource.c_str();
    
    GLuint vertShader, fragShader;
    
    program = glCreateProgram();
    
    vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertShaderSource, NULL);
    glCompileShader(vertShader);
    cerr << "Compiling " << vertProgFilename << "." << endl;
    printShaderInfoLog(vertShader);
    
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragShaderSource, NULL);
    glCompileShader(fragShader);
    cerr << "Compiling " << fragProgFilename << "." << endl;
    printShaderInfoLog(fragShader);
    
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);
    cerr << "Enabling program." << endl;
    printProgramInfoLog(program);
    glUseProgram(program);
    checkGLErrors("End of readShader");
}

//--------------------------------------------------------------------------
// Runs the current shader program/texture state/render target across
// a fullscreen quad.  Primarily useful for post-processing or GPGPU.
//--------------------------------------------------------------------------
void renderFullscreenQuad()
{
    // Make the screen go from 0,1 in the x and y direction, with no
    // frustum effect (that is, increasing z doesn't shrink x and y).
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1);
    
    // Don't do any model transformation.
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    // Draw a fullscreen quad with appropriate tex coords.
    glBegin(GL_POLYGON);
    glTexCoord2f(0, 0);
    glVertex3f(0, 0, 0);
    
    glTexCoord2f(0, 1);
    glVertex3f(0, 1, 0);
    
    glTexCoord2f(1, 1);
    glVertex3f(1, 1, 0);
    
    glTexCoord2f(1, 0);
    glVertex3f(1, 0, 0);
    glEnd();
    
    // Restore the modelview matrix.
    glPopMatrix();
    
    // Restore the projection matrix.
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    
    // Always good practice to get back to modelview mode at all times.
    glMatrixMode(GL_MODELVIEW);
    checkGLErrors("renderFullscreenQuad");
}
