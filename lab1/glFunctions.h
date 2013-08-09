// Sets up initial status of OpenGL.
void initGL();
// Sets up an OpenGL light.  This only needs to be called once
// and the light will be used during all renders.
void initLights();
// Sets the OpenGL material state.  This is remembered so we only need to
// do this once.  If you want to use different materials, you'd need to do this
// before every different one you wanted to use.
void initMaterial();
// Check OpenGL error status, and if an error is encountered, prints the error
// and label.
void checkGLErrors(const char *label);
// Initializes the texture with the supplied data as starting values.
void createTexture(GLuint &texNum, GLvoid *pixels, int width, int height,
                   GLenum internalFormat=GL_RGB, GLenum uploadType=GL_UNSIGNED_BYTE,
                   GLenum uploadFormat=GL_RGB);
// Initializes a texture from a png file.
void createTexture(GLuint &texNum, const char *pngFilename,
                   GLenum internalFormat=GL_RGB);
// Prints an info log regarding the creation of a vertex or fragment shader
void printShaderInfoLog(GLuint obj);
// Prints an info log regarding the creation of an entire GPU program
void printProgramInfoLog(GLuint obj);
// Reads the shader at the supplied filenames, and compiles it into program
void readShader(const string vertProgFilename,
                const string fragProgFilename,
                GLuint &program);
// Runs the current shader program/texture state/render target across
// a fullscreen quad.  Primarily useful for post-processing or GPGPU.
void renderFullscreenQuad();
