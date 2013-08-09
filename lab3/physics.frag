uniform sampler2DRect positionTex;
uniform sampler2DRect velocityTex;
uniform float prand;
uniform float rand;

const float h = .03f;
const float gravConstant = .1; //make sure same gravConstant as in energy.frag
const float largeGravConstantFactor = 5000.; //make sure same largeGravConstantFactor as in energy.frag
const vec3 center  = vec3(.5, .5, .5); //make sure same center as in energy.frag
const vec3 maxVec = vec3(2.f, 2.f, 2.f); 
const float deathPerIteration = 0.005;
const float smallGravityRadiusThreshold = 0.64; //make sure same smallGravityRadiusThreshold as in energy.frag

vec3 smallForce(vec3 loc);
vec3 largeForce(vec3 loc);
vec3 clipToReasonableVelocity(vec3 velocity);

void main() {
    vec2 texCoord = gl_TexCoord[0].xy;
    vec3 position = texture2DRect(positionTex, texCoord.xy).xyz;
    vec3 velocity = texture2DRect(velocityTex, texCoord.xy).xyz;
    float life = texture2DRect(positionTex, texCoord.xy).w;

    if (life < 0.f)
    {
    	// a reasonable way to generate a random number between 0 and 1
		float random = cos(3.141526*(position.x*100.*rand + position.y*100.*rand + position.z*100.*rand));
		float theta = 2.*3.1415926*random;
		position = vec3(cos(theta)+.5f, sin(theta)+.5f, 0.);
		velocity = vec3(0., 0., 0.);
		life = random;
    }

    // : calculate new position and velocity.  Since this is a MRT shader, must render to
      //gl_FragData[0] (velocity) and gl_FragData[1] (position).

    float radius = length(center - position);
    if (radius < smallGravityRadiusThreshold)
    {
  	    velocity += h*smallForce(position);
    }
    else
    {
    	velocity += h*largeForce(position);
    }
   	velocity = clipToReasonableVelocity(velocity);
    position += h*velocity;
    life -= deathPerIteration;

    gl_FragData[0] = vec4(velocity.xyz, 1.);
    gl_FragData[1] = vec4(position.xyz, life);
}

vec3 smallForce(vec3 loc)
{
	return normalize(center - loc) * gravConstant * 1.f/pow(length(center - loc), 2);
}

vec3 largeForce(vec3 loc)
{
	return normalize(center - loc) * (gravConstant * largeGravConstantFactor) * 1.f/pow(length(center - loc), 2);
}

vec3 clipToReasonableVelocity (vec3 velocity)
{
	if (velocity.x > maxVec.x)
	{
		velocity.x = maxVec.x;
	}
	if (velocity.x < -1.*maxVec.x)
	{
		velocity.x = maxVec.x*-1.;
	}

	if (velocity.y > maxVec.y)
	{
		velocity.y = maxVec.y;
	}
	if (velocity.y < -1.*maxVec.y)
	{
		velocity.y = maxVec.y*-1.;
	}

	if (velocity.z > maxVec.z)
	{
		velocity.z = maxVec.z;
	}
	if (velocity.z < -1.*maxVec.z)
	{
		velocity.z = maxVec.z*-1.;
	}
	return velocity;
}