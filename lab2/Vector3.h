#ifndef	__VECTOR3_H__
#define	__VECTOR3_H__

#include <iostream>
#include <stddef.h>
#include <math.h>
#include "util.h"

// column type 3 vectors
class Vector3{
public:
    enum{
        X = 0,
        Y = 1,
        Z = 2
    };
    Vector3( void ) { m_v[X] = m_v[Y] = m_v[Z] = 0; }
    Vector3( const Vector3& c )
    { m_v[X] = c.m_v[X], m_v[Y] = c.m_v[Y], m_v[Z] = c.m_v[Z]; }
    Vector3( const float a, const float b, const float c )
    { m_v[X] = a, m_v[Y] = b, m_v[Z] = c; }
    Vector3( const float* a )
    { m_v[X] = a[X], m_v[Y] = a[Y], m_v[Z] = a[Z]; }
    Vector3( const double* a )
    { m_v[X] = float( a[X] ), m_v[Y] = float( a[Y] ), m_v[Z] = float( a[Z] ); }
    operator float*( void )
    { return &m_v[X]; }
    operator const float*( void ) const
    { return &m_v[X]; }
    
    friend std::istream& operator>>( std::istream&, Vector3& );
    friend std::ostream& operator<<( std::ostream&, const Vector3& );
    void print( std::ostream& os ) const;
    bool read ( std::istream& is );
    
    Vector3& operator=( const Vector3& c )
    { m_v[X] = c.m_v[X], m_v[Y] = c.m_v[Y], m_v[Z] = c.m_v[Z]; return *this; }
    Vector3 operator+( const Vector3& c ) const
    { return Vector3( m_v[X] + c.m_v[X], m_v[Y] + c.m_v[Y], m_v[Z] + c.m_v[Z] ); }
    Vector3 operator-( const Vector3& c ) const
    { return Vector3( m_v[X] - c.m_v[X], m_v[Y] - c.m_v[Y], m_v[Z] - c.m_v[Z] ); }
    Vector3 operator*( const float s ) const
    { return Vector3( s * m_v[X], s * m_v[Y], s * m_v[Z] ); }
    friend Vector3 operator*( const float s, const Vector3& c )
    { return Vector3( s * c.m_v[X], s * c.m_v[Y], s * c.m_v[Z] ); }
    Vector3 operator/( const float s ) const
    { return Vector3( m_v[X] / s, m_v[Y] / s, m_v[Z] / s ); }
    
    Vector3& operator+=( const Vector3& c )
    { m_v[X] += c.m_v[X], m_v[Y] += c.m_v[Y], m_v[Z] += c.m_v[Z]; return *this; }
    Vector3& operator-=( const Vector3& c )
    { m_v[X] -= c.m_v[X], m_v[Y] -= c.m_v[Y], m_v[Z] -= c.m_v[Z]; return *this; }
    Vector3& operator*=( const float s )
    { m_v[X] *= s, m_v[Y] *= s, m_v[Z] *= s; return *this; }
    Vector3& operator/=( const float s )
    { m_v[X] /= s, m_v[Y] /= s, m_v[Z] /= s; return *this; }
    Vector3 operator-( void ) const
    { return Vector3( -m_v[X], -m_v[Y], -m_v[Z] ); }
    
    float& x( void ) { return m_v[X]; }
    float x( void ) const { return m_v[X]; }
    float& y( void ) { return m_v[Y]; }
    float y( void ) const { return m_v[Y]; }
    float& z( void ) { return m_v[Z]; }
    float z( void ) const { return m_v[Z]; }
    float& operator() (const int i)      { return m_v[i]; }
    float operator() (const int i) const { return m_v[i]; }
    
    Vector3 lerp( const Vector3& v1, const float t ) const{
        return Vector3( m_v[X] + t * ( v1.m_v[X] - m_v[X] ),
                     m_v[Y] + t * ( v1.m_v[Y] - m_v[Y] ),
                     m_v[Z] + t * ( v1.m_v[Z] - m_v[Z] ) );
    }
    
    Vector3 min( const Vector3& o ) const{
        const float a = std::min( m_v[X], o.m_v[X] );
        const float b = std::min( m_v[Y], o.m_v[Y] );
        const float c = std::min( m_v[Z], o.m_v[Z] );
        return Vector3( a, b, c );
    }
    Vector3 max( const Vector3& o ) const{
        const float a = std::max( m_v[X], o.m_v[X] );
        const float b = std::max( m_v[Y], o.m_v[Y] );
        const float c = std::max( m_v[Z], o.m_v[Z] );
        return Vector3( a, b, c );
    }
    void bbox( Vector3& min, Vector3& max ){
        if( m_v[X] < min.m_v[X] ) min.m_v[X] = m_v[X];
        else if( m_v[X] > max.m_v[X] ) max.m_v[X] = m_v[X];
        if( m_v[Y] < min.m_v[Y] ) min.m_v[Y] = m_v[Y];
        else if( m_v[Y] > max.m_v[Y] ) max.m_v[Y] = m_v[Y];
        if( m_v[Z] < min.m_v[Z] ) min.m_v[Z] = m_v[Z];
        else if( m_v[Z] > max.m_v[Z] ) max.m_v[Z] = m_v[Z];
    }
    
    float dot( const Vector3& c ) const
    { return m_v[X] * c.m_v[X] + m_v[Y] * c.m_v[Y] + m_v[Z] * c.m_v[Z]; }
    float dot( void ) const
    { return m_v[X] * m_v[X] + m_v[Y] * m_v[Y] + m_v[Z] * m_v[Z]; }
    Vector3 cross( const Vector3& c ) const
    { return Vector3( m_v[Y] * c.m_v[Z] - m_v[Z] * c.m_v[Y],
                   m_v[Z] * c.m_v[X] - m_v[X] * c.m_v[Z],
                   m_v[X] * c.m_v[Y] - m_v[Y] * c.m_v[X] ); }
    float l1( void ) const
    { float a = abs( m_v[X] ); a += abs( m_v[Y] ); return a + abs( m_v[Z] ); }
    float linfty( void ) const{
        float a = abs( m_v[X] ); a = std::max( a, abs( m_v[Y] ) );
        return std::max( a, abs( m_v[Z] ) );
    }
    float l2( void ) const { return sqrtf( dot() ); }
    
    Vector3 dir( void ) const;
    
    int normalize( void )
    { float mag = l2(); return zero( mag ) ? 0 : ( *this *= 1 / mag, 1 ); }
    float dist( const Vector3& c ) const { return ( *this - c ).l2(); }
    
protected:
private:
        float	m_v[3];
};

typedef Vector3* Vector3p;

#endif	/* __VECTOR3_H__ */
