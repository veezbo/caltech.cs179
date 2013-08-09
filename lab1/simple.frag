varying vec3 normal, lightDir, worldpos;

void main()
{
    vec4 color = gl_LightModel.ambient * gl_FrontMaterial.ambient;

    vec3 N, L, pos;
    N = normalize(normal);
    L = normalize(lightDir);
    pos = normalize(worldpos);

    float NdotL = max(dot(N, L), 0.);
   
    //so we're actually just calculating the angle between the normal and light direction
    //and this happens implicity since N and L are normalized

    if (NdotL > 0.0)
    {
   		color += (gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse * NdotL);
    }

    vec4 Cs = gl_FrontMaterial.specular * gl_LightSource[0].specular;
    float REye = max(dot( -reflect(L, N), -1.0f * pos), 0.0);
    float S = gl_FrontMaterial.shininess;
    color += Cs * pow(REye, S);
	
    gl_FragColor = color;
}
