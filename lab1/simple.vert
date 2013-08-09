varying vec3 normal, lightDir, worldpos;

void main()	
{
    normal = normalize(gl_NormalMatrix * gl_Normal);
    worldpos = vec3(gl_ModelViewMatrix * gl_Vertex);
    lightDir = normalize(vec3(gl_LightSource[0].position) - worldpos);
    gl_Position = ftransform();
    //gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;

}
