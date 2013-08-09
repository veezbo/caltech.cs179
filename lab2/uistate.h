// -*- Mode: c++ -*-
// $Id: uistate.h,v 1.2 2004/01/26 18:03:29 ps Exp $
// $Source: /cs/research/multires/cvsroot/src/frame/uistate.h,v $

#ifndef	__UISTATE_H__
#define	__UISTATE_H__

// Do not change anything above this line

// include files
#include <assert.h>
#include "ArcBall.h"
#include "Vector3.h"

typedef void (*vpfv)( void );
typedef void (*vpfii)( const int x, const int y );

class UIState{
public:
    UIState( void );
    ~UIState( void ) {}
    
    // accessors
    int& MouseX( void ) { return p_mousex; }
    int& MouseY( void ) { return p_mousey; }
    int& WindowX( void ) { return p_windowx; }
    int& WindowY( void ) { return p_windowy; }
    float& Radius( void ) { return p_radius; }
    Vector3& Trans( void ) { return p_trans; }
    Vector3& CTrans( void ) { return p_ctrans; }
    Vector3& COldTrans( void ) { return p_coldtrans; }
    ArcBall& Ball( void ) { return p_ball; }
    float& Aspect( void ) { return p_aspect; }
    float& Near( void ) { return p_near; }
    float& Far( void ) { return p_far; }
    float& Fov( void ) { return p_fov; }
    void MouseFunction(const int button, const int state, const int x, const int y);
    void MotionFunction(const int x, const int y);
    enum{
        LEFT_BUTTON = 1,
        MIDDLE_BUTTON = 2,
        RIGHT_BUTTON = 4
    };
    int ButtonL( void ) { return p_button & LEFT_BUTTON; }
    int ButtonM( void ) { return p_button & MIDDLE_BUTTON; }
    int ButtonR( void ) { return p_button & RIGHT_BUTTON; }
    
    // drawing styles
    enum DrawStyle{
        HIDDENLINE,
        FLATSHADED,
        SMOOTHSHADED,
        VISUALIZE
    };
    DrawStyle DStyle( void ) { return p_dstyle; }
    void SetDstyle( DrawStyle s ) {
        assert( s == HIDDENLINE || s == FLATSHADED ||
                s == SMOOTHSHADED || s == VISUALIZE );
        p_dstyle = s;
    }
    int& DirtyFlag( void ) { return p_dlistDirty; }
    
    // methods
    void ResetModelTransform( void );
    void ApplyViewingTransformation( void );
    void SetupViewingFrustum( void );
    void SetupViewport( void );
    
    void OnLButtonDown( const int x, const int y );
    void OnLButtonUp( const int x, const int y );
    void OnMButtonDown( const int x, const int y );
    void OnMButtonUp( const int x, const int y );
    void OnRButtonDown( const int x, const int y );
    void OnRButtonUp( const int x, const int y );
    
protected:
private:
    // ui state
    // last known mouse position
    int p_mousex, p_mousey;
    // last known window size
    int p_windowx, p_windowy;
    // which one is currently pressed?
    int p_button;
    // what style of drawing?
    DrawStyle p_dstyle;
    int p_dlistDirty;
    
    // radius of the world
    float p_radius;
    // translation due to centroid of object
    Vector3 p_trans;
    // camera translation to get a good view
    Vector3 p_ctrans;
    Vector3 p_coldtrans;
    
    // camera parameters
    float p_aspect;
    float p_near;
    float p_far;
    float p_fov;
    
    // arcball interface
    ArcBall p_ball;
};

// extern defs for non-member outline functions go here
// externally visible constant defs go here
#include "util.h"

inline const float Deg2Rad( const float d ){
    return (float(M_PI)/180.f)*d;
}

// Initializes the UI
void uiInit();

#endif	/* __UISTATE_H__ */
