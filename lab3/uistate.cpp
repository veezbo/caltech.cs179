// -*- Mode: c++ -*-
// $Id: uistate.cpp,v 1.3 2004/01/31 03:19:19 ps Exp $
// $Source: /cs/research/multires/cvsroot/src/frame/uistate.cpp,v $

// include files
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "uistate.h"
#include "util.h"

// code

UIState::UIState( void )
: p_mousex(-1), p_mousey(-1),
p_windowx(-1), p_windowy(-1),
p_button(0),
p_dstyle(HIDDENLINE),
p_radius( 1 ),
p_trans(0,0,0),
p_ctrans( 0, 0, -( 1 + 1 / sinf( Deg2Rad( 25 ) ) ) ),
p_coldtrans(0,0,0),
p_aspect( 1 ),
p_near( 1 / sinf( Deg2Rad( 25 ) ) ),
p_far( 1 / sinf( Deg2Rad( 25 ) ) + 2 ),
p_fov( 50 ),
p_ball()
{
}

void
UIState::SetupViewport( void )
{
    glViewport( 0, 0, p_windowx, p_windowy );
}

void
UIState::SetupViewingFrustum( void )
{
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( p_fov / std::min( 1.f, p_aspect ), p_aspect, p_near, p_far );
    glMatrixMode( GL_MODELVIEW );
}

void
UIState::ApplyViewingTransformation( void )
{
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( p_ctrans.x(), p_ctrans.y(), p_ctrans.z() );
    
    GLfloat M[4][4];
    p_ball.Update();
    p_ball.Value( M );
    // apply model transform
    glMultMatrixf( ( float *)M );
    // move world to the origin
    glTranslatef( -p_trans.x(), -p_trans.y(), -p_trans.z() );
}

void
UIState::OnLButtonDown( const int x, const int y )
{
    p_button |= LEFT_BUTTON;
    p_ball.Mouse( ( p_aspect>1 ? p_aspect : 1 ) * ( 2.f*x / p_windowx - 1 ),
                  -( p_aspect>1 ? 1 : 1/p_aspect ) * ( 2.f*y / p_windowy - 1 ) );
    switch( glutGetModifiers() ){
        case GLUT_ACTIVE_SHIFT:
            p_ball.UseSet( ArcBall::CameraAxes );
            p_ball.Update();
            glutPostRedisplay();
            break;
        case GLUT_ACTIVE_CTRL:
            p_ball.UseSet( ArcBall::BodyAxes );
            p_ball.Update();
            glutPostRedisplay();
            break;
        case GLUT_ACTIVE_ALT:
            break;
        default:
            ;
    }
    p_ball.BeginDrag();
}

void
UIState::OnLButtonUp( const int, const int )
{
    p_button &= ~LEFT_BUTTON;
    p_ball.EndDrag();
    p_ball.UseSet( ArcBall::NoAxes );
    p_ball.Update();
    glutPostRedisplay();
}

void
UIState::OnMButtonDown( const int x, const int y )
{
    p_button |= MIDDLE_BUTTON;
    p_ball.Mouse( ( p_aspect>1 ? p_aspect : 1 ) * ( 2.f*x / p_windowx - 1 ),
                  -( p_aspect>1 ? 1 : 1/p_aspect ) * ( 2.f*y / p_windowy - 1 ) );
    p_ball.BeginTrans();
    p_coldtrans = p_ctrans;
}

void
UIState::OnMButtonUp( const int, const int )
{
    p_button &= ~MIDDLE_BUTTON;
    glutPostRedisplay();
}

void
UIState::OnRButtonDown( const int x, const int y )
{
    p_button |= RIGHT_BUTTON;
    p_ball.Mouse( ( p_aspect>1 ? p_aspect : 1 ) * ( 2.f*x / p_windowx - 1 ),
                  -( p_aspect>1 ? 1 : 1/p_aspect ) * ( 2.f*y / p_windowy - 1 ) );
    p_ball.BeginTrans();
    p_coldtrans = p_ctrans;
}

void
UIState::OnRButtonUp( const int, const int )
{
    p_button &= ~RIGHT_BUTTON;
    glutPostRedisplay();
}

void
UIState::ResetModelTransform( void )
{
    p_ctrans = Vector3( 0, 0, -p_radius * ( 1.f + 1 / sinf( Deg2Rad( p_fov/2 ) ) ) ),
    p_ball.Reset();
}

void UIState::MouseFunction(int button, int state, int x, int y)
{
    switch( button ){
        case GLUT_LEFT_BUTTON:
            if( state == GLUT_DOWN ) OnLButtonDown( x, y );
            else if( state == GLUT_UP ) OnLButtonUp( x, y );
            break;
        case GLUT_MIDDLE_BUTTON:
            if( state == GLUT_DOWN ) OnMButtonDown( x, y );
            else if( state == GLUT_UP ) OnMButtonUp( x, y );
            break;
        case GLUT_RIGHT_BUTTON:
            if( state == GLUT_DOWN ) {
                OnRButtonDown( x, y );
                p_mousex = x;
                p_mousey = y;
            }
            else if( state == GLUT_UP ) OnRButtonUp( x, y );
            break;
        default:
            break;
    }
}

void
UIState::MotionFunction(const int x, const int y)
{
    Ball().Mouse( ( Aspect() > 1 ? Aspect() : 1 )
                      * ( 2.f * x / WindowX() - 1 ),
                      -( Aspect() > 1 ? 1 : 1/Aspect() )
                      * ( 2.f * y / WindowY() - 1 ) );
    if( ButtonM() ){
        // translation case
        Vector3 t = Ball().Trans();
        CTrans().z() = ( 1 - 2 * t.y() ) * COldTrans().z();
    }
    if( ButtonR() ){
        CTrans().x() += .005 * (float)(x - p_mousex);
        CTrans().y() += .005 * (float)(p_mousey - y);
        p_mousex = x;
        p_mousey = y;
    }
    glutPostRedisplay();
}
