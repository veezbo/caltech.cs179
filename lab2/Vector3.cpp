#include <math.h>
#include <assert.h>

#include "Vector3.h"

void 
Vector3::print(std::ostream& os) const
{
  os << m_v[0] << " " << m_v[1] << " " << m_v[2];
}

std::ostream&
operator<<( std::ostream& os, const Vector3& c )
{
  c.print( os );	
  return os;
}

std::istream&
operator>>( std::istream& is, Vector3& f )
{
  return is >> f.m_v[0] >> f.m_v[1] >> f.m_v[2];
}


bool
Vector3::read( std::istream& is )  {
  return ( is >> m_v[0]) && (is >> m_v[1]) && (is  >> m_v[2]);
}

Vector3
Vector3::dir( void ) const
{
  float a = l2();
  if( zero( a ) ) return *this;
  else return *this / a;
}
