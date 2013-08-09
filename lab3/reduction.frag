uniform sampler2DRect reductionTex;

const float w = 256., h = 256.; //make sure these values are the same as in main.cpp

void main()
{
	vec2 texCoord = gl_TexCoord[0].xy;
	float wCoord = texCoord.x;
	float hCoord = texCoord.y;

    // : Reduce size of texture by half by averaging four surrounding components.
    float val = 0.;
    float ll = texture2DRect(reductionTex, vec2(2.*wCoord, 2.*hCoord)).x;
    float lr = texture2DRect(reductionTex, vec2(2.*wCoord+1., 2.*hCoord)).x;
    float ul = texture2DRect(reductionTex, vec2(2.*wCoord, 2.*hCoord+1.)).x;
    float ur = texture2DRect(reductionTex, vec2(2.*wCoord+1., 2.*hCoord+1.)).x;

    val += (ll + lr + ul + ur);
    val = abs(val);

	gl_FragData[0] = vec4(val*0.25, 0., 0., 1.);
}