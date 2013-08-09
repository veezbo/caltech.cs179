uniform sampler2D bumpTex, colorTex;

varying vec3 eyeDir,lightDir;

uniform int shadowsOn;

const float bumpScale = 0.07;
const float nSteps = 100.; //should be an int regardless of type

// : Calculate normal. You can use the same code as in bump.frag
vec3 getNormal(float x, float y)
{
    vec3 normal = vec3(0, 0, 1);

    // : Calculate normal from sampling bumpTex
    vec3 tangent, binormal;
    const float dt = 0.0009765625f;
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
    vec3 L = normalize(lightDir);
    vec3 E = normalize(eyeDir);
    // : Raytrace back along the eye direction to calculate where we
    // intersect the bump surface.
    float step = 1.f/nSteps;
    vec2 delta = -E.xy * bumpScale/(E.z*nSteps);
    float height = 1.0;
    vec2 t = gl_TexCoord[0].st;
    float bump = texture2D(bumpTex, t).r;
    float count = 0.; //should still be an integer

    while (bump < height && count < nSteps)
    {
        height -= step;
        t += delta;
        count++;
        bump = texture2D(bumpTex, t).r;
    }
    vec3 normal = getNormal(t.x, t.y); //normal for diffuse lighting calculation later

    // : Raytrace to get shadows
    int inShadow = 0;
    if (shadowsOn != 0)
    {
        vec2 shadowDelta = L.xy * bumpScale/(L.z*nSteps);
        vec2 shadowT = t; //start at the final coord found earlier
        float shadowBump = bump;
        float shadowHeight = texture2D(bumpTex, shadowT).r + step*0.1;

        while (shadowBump < shadowHeight && shadowHeight < 1.)
        {
            shadowHeight += step;
            shadowT += shadowDelta;
            shadowBump = texture2D(bumpTex, shadowT).r;
        }
        if (shadowBump >= shadowHeight)
        {
            inShadow = 1;
        }
    }


    // : Replace gl_FrontMaterial with appropriate color sampling,
    // and make the color darker if shadowsOn and this fragment is in shadow.
    // Also replace normal with appropriate call to getNormal.
    vec4 sampleTexColor = texture2D(colorTex, t.xy).rgba;
    //vec4 color = gl_LightModel.ambient * gl_FrontMaterial.ambient;
    vec4 color = gl_LightModel.ambient * sampleTexColor;
    
    //vec3 normal = vec3(0, 0, 1);
    float NdotL = max(dot(normalize(normal),normalize(lightDir)),0.0);
    if (NdotL > 0.0)
    {
        //color += (gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse * NdotL); 
        color += (sampleTexColor * gl_LightSource[0].diffuse * NdotL);
    }
    if (inShadow == 1)
    {
        gl_FragColor = color * 0.1;
    }
    else
    {
        gl_FragColor = color;
    }
   
}
