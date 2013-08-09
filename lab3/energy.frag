uniform sampler2DRect positionTex;
uniform sampler2DRect velocityTex;

const vec3 center = vec3(.5, .5, .5); //make sure same center as in physics.frag
const float gravConstant = .1f; //make sure same gravConstant as in physics.frag
const float largeGravConstantFactor = 5000.; //make sure same largeGravConstantFactor as in physics.frag
const float smallGravityRadiusThreshold = 0.64; //make sure same smallGravityRadiusThreshold as in physics.frag

void main()
{
    vec2 texCoord = gl_TexCoord[0].xy;
    vec3 position = texture2DRect(positionTex, texCoord.xy).xyz;
    vec3 velocity = texture2DRect(velocityTex, texCoord.xy).xyz;

    // :  Calculate particle energy.
    float energy = 0.;
    float potentialEnergy = 0., kineticEnergy = 0.;
    float radius = length(center - position);
    if (radius < smallGravityRadiusThreshold)
    {
    	potentialEnergy = gravConstant * -1.f/radius;
    }
    else
    {
	   	potentialEnergy = gravConstant * -1.f/smallGravityRadiusThreshold;
    	potentialEnergy += gravConstant * largeGravConstantFactor * -1.f/(radius - smallGravityRadiusThreshold);
    }

    kineticEnergy = 0.5*dot(velocity, velocity);
    energy = potentialEnergy + kineticEnergy;
    
    gl_FragData[0] = vec4(energy);
}