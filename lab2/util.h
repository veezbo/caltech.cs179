// -*- Mode: c++ -*-
// $Id: util.h,v 1.1.1.1 2004/01/18 02:59:41 ps Exp $
// $Source: /cs/research/multires/cvsroot/src/frame/util.h,v $

#ifndef	__UTIL_H__
#define	__UTIL_H__

#include <iostream>
#include <assert.h>

#ifdef _WIN32
#define fatan2(x,y) atan2(x,y)
#endif

#ifdef _WIN32
#include <float.h>
#define isnanf(x) (_isnan(x))
#endif

// Inline function definitions go here
template <class _type>
inline int
argmin( _type a, _type b, _type c )
{
  return a <= b ? ( a <= c ? 0 : 2 ) : ( b <= c ? 1 : 2 );
}

template <class _type>
inline _type
clip( _type value, _type lower, _type upper )
{
  return std::min( std::max( value, lower ), upper );
}

template <class _type>
inline _type
min( _type a, _type b, _type c )
{
  const _type m = std::min( a, b );
  return std::min( m, c );
}

template <class _type>
inline _type
max( _type a, _type b, _type c )
{
  const _type m = std::max( a, b );
  return std::max( m, c );
}

template <class _type>
inline _type
min( _type a, _type b, _type c, _type d )
{
  _type m = std::min( a, b );
  m = std::min( m, c );
  return std::min( m, d );
}

template <class _type>
inline _type
max( _type a, _type b, _type c, _type d )
{
  _type m = std::max( a, b );
  m = std::max( m, c );
  return std::max( m, d );
}

template <class _type>
inline int
sign( const _type f )
{
  return f < 0 ? -1 : ( f > 0 ? 1 : 0 );
}

template <class _type>
inline _type
abs( const _type f )
{
  return f < 0 ? -f : f;
}

template <class _type>
inline _type
sqr( const _type s )
{
  return s * s;
}

template <class _type>
inline _type
cube( const _type s )
{
  return s * s * s;
}

inline int
even( int i )
{
  return !( i % 2 );
}

inline int
odd( int i )
{
  return i % 2;
}

inline int
zero( const float x, const float eps )
{
  return ( -eps < x ) && ( x < eps );
}

static float __negeps_f = -1e-6f;
static float __poseps_f =  1e-6f;

inline int
zero( const float x )
{
  return ( __negeps_f < x ) && ( x < __poseps_f );
}

// disable VC6.0 warning "identifier was truncated to 255 characters "
// this is routine when compiling templates
// unfortunately, it does not seem to work in all cases
// and the messages are still visible when compiling STL

#if defined(_WIN32) && !defined(__GNUG__)
#pragma warning(disable:4786)
#define M_PI		3.14159265358979323846
#endif

#include <assert.h>
// a special version with diagnostics
// modeled after assert.h
#ifdef __cplusplus
extern "C" {
#endif
#ifdef NDEBUG

#undef assertzero
#define assertzero(EX) ((void)0)
#define die() ((void)0)
#else

#if defined(_WIN32) && !defined(__GNUG__)
#define asse _assert
#else
#define asse __assert
#endif

#ifdef __ANSI_CPP__
#define assertzero(EX)  ((zero(EX))?((void)0):\
			 (cerr.precision(12),\
			  cerr << "assertion on zero( " <<\
			  # EX << " failed by " <<\
			  (EX)/m_eps_f << " eps\n",\
			 asse( # EX , __FILE__, __LINE__)))
#else
#define assertzero(EX)  ((zero(EX))?((void)0):\
			 (cerr.precision(12),\
			  cerr << "assertion on zero( " <<\
			  "EX" << " failed by " <<\
			  (EX)/m_eps_f << " eps\n",\
			 asse( "EX", __FILE__, __LINE__)))
#endif

#define die()            (asse( "disallowed", __FILE__, __LINE__ ))

#endif /* NDEBUG */

#ifdef __cplusplus
}
#endif

#endif	/* __UTIL_H__ */
