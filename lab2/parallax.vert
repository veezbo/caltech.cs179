#version 120    // Necessary for clean compile using transpose() function.
                // If your card doesn't support, take out, and do transpose
                // yourself.

uniform sampler2D bumpTex, colorTex;

varying vec3 lightDir, eyeDir;

attribute vec3 tangent;

// Note: This file can be the same as bump.vert
void main()
{
    // : Calculate TBN
    vec3 T, B, N;
    N = normalize(gl_NormalMatrix * gl_Normal);
    T = normalize(gl_NormalMatrix * tangent);
    B = cross(N, T);

    // : Use TBN to put eyeDir, lightDir in surface coords

    // First, put lightDir and eyeDir in world coordinates
    eyeDir = vec3( gl_ModelViewMatrix * gl_Vertex ); //same as worldpos
    lightDir = normalize(gl_LightSource[0].position.xyz - eyeDir);
    
    // Now, convert to surface coordinates using the TBN matrix
    eyeDir = vec3(dot(T, eyeDir), dot(B, eyeDir), dot(N, eyeDir));
    lightDir = vec3(dot(T, lightDir), dot(B, lightDir), dot(N, lightDir));

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = ftransform();
}
