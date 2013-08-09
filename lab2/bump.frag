#version 120

uniform sampler2D bumpTex, colorTex;

varying vec3 eyeDir,lightDir;

uniform int shadowsOn;

vec3 getNormal(float x, float y) {
    vec3 normal = vec3(0, 0, 1);
    
    // : Calculate normal from sampling bumpTex
    vec3 tangent, binormal;
    const float dt = 1/1024.;
    const float C = 35.;
    float N, S, E, W;
    N = texture2D(bumpTex, vec2(x, y+dt)).r;
    S = texture2D(bumpTex, vec2(x, y-dt)).r;
    W = texture2D(bumpTex, vec2(x-dt, y)).r;
    E = texture2D(bumpTex, vec2(x+dt, y)).r;

    tangent = normalize(vec3(C*dt, 0, E - W));
    binormal = normalize(vec3(0, C*dt, N - S));
    normal = cross(tangent, binormal);
    return normal;
}
void main()
{
    // : replace gl_FrontMaterial ambient with texture sampling
    float texX, texY;
    texX = gl_TexCoord[0].s;
    texY = gl_TexCoord[0].t;

    // Sampling the Color Texture at TexCoords to get the color
    vec4 sampleTexColor = texture2D(colorTex, vec2(texX, texY)).rgba;
    //vec4 color = gl_LightModel.ambient * gl_FrontMaterial.ambient;
    vec4 color = gl_LightModel.ambient * sampleTexColor;
    vec3 normal = getNormal(texX, texY);

    float NdotL = max(dot(normalize(normal),normalize(lightDir)),0.0);
    if (NdotL > 0.0)
    {
        //color += (gl_FrontMaterial.ambient * gl_LightSource[0].diffuse * NdotL); 
        color += (sampleTexColor * gl_LightSource[0].diffuse * NdotL);
    }

    gl_FragColor = color;
}
