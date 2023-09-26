// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "owl/common/math/box.h"
#include "owl/common/math/LinearSpace.h"

#include <vector>
#include <memory>
#ifdef _GNUC_
#include <unistd.h>
#endif

inline float toRadian(float deg) { return deg * float(M_PI / 180.f); }
inline float toDegrees(float rad) { return rad / float(M_PI / 180.f); }

struct OWLViewer;
using namespace owl;

namespace dtracker
{
/*! the entire state for someting that can 'control' a camera -
    ie, that can rotate, move, focus, force-up, etc, a
    camera... for which it needs way more information than the
    simple camera.

    Note this uses a RIGHT HANDED camera as follows:
    - logical "up" is y axis
    - right is x axis
    - depth is _NEGATIVE_ z axis
*/
struct Camera
{
    Camera()
    {}

    /*! get the point of interest */
    vec3f getPOI() const { return position - poiDistance * frame.vz; }
    /*! get field of view (degrees)*/
    inline float getFovyInDegrees() const { return fovyInDegrees; }
    /*! get cos of field of view */
    inline float getCosFovy() const { return cosf(toRadian(fovyInDegrees)); }
    /*! get camera eye position*/
    inline vec3f getFrom() const { return position; }
    /*! get camera center */
    inline vec3f getAt() const { return position - frame.vz; }
    /*! get camera up vector */
    inline vec3f getUp() const { return frame.vy; }

    /*! set field of view (degrees)*/
    inline void setFovy(const float fovy) { fovyInDegrees = fovy; }
    /*! set camera focal distance*/
    inline void setFocalDistance(float dist) { this->focalDistance = dist; }
    /*! set given aspect ratio */
    inline void setAspect(const float aspect) { this->aspect = aspect; }
    /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
    inline void forceUpFrame()
    {
        // frame.vz remains unchanged
        if (fabsf(dot(frame.vz, upVector)) < 1e-6f)
            // looking along upvector; not much we can do here ...
            return;
        frame.vx = normalize(cross(upVector, frame.vz));
        frame.vy = normalize(cross(frame.vz, frame.vx));
    }
    /*! re-compute all orientation related fields from given
        'user-style' camera parameters */
    inline void setOrientation(/* camera origin    : */ const vec3f &origin,
                        /* point of interest: */ const vec3f &interest,
                        /* up-vector        : */ const vec3f &up,
                        /* fovy, in degrees : */ float fovyInDegrees,
                        /* set focal dist?  : */ bool setFocalDistance = true)
    {
        this->fovyInDegrees = fovyInDegrees;
        position = origin;
        upVector = up;
        frame.vz = (interest == origin)
                    ? vec3f(0, 0, 1)
                    : /* negative because we use NEGATIZE z axis */ 
                    -normalize(interest - origin);
        frame.vx = cross(up, frame.vz);
        if (dot(frame.vx, frame.vx) < 1e-8f)
            frame.vx = vec3f(0, 1, 0);
        else
            frame.vx = normalize(frame.vx);
        // frame.vx
        //   = (fabs(dot(up,frame.vz)) < 1e-6f)
        //   ? vec3f(0,1,0)
        //   : normalize(cross(up,frame.vz));
        frame.vy = normalize(cross(frame.vz, frame.vx));
        poiDistance = length(interest - origin);
        if (setFocalDistance)
            focalDistance = poiDistance;
        forceUpFrame();
    }

    inline void setUpVector(const vec3f &up)
    {
        upVector = up;
        forceUpFrame();
    }

    linear3f frame{one};
    vec3f position{0, -1, 0};
    /*! distance to the 'point of interst' (poi); e.g., the point we
        will rotate around */
    float poiDistance{1.f};
    float focalDistance{1.f};
    vec3f upVector{0, 1, 0};
    /* if set to true, any change to the frame will always use to
       upVector to 'force' the frame back upwards; if set to false,
       the upVector will be ignored */
    bool forceUp{true};

    /*! multiplier how fast the camera should move in world space
        for each unit of "user specifeid motion" (ie, pixel
        count). Initial value typically should depend on the world
        size, but can also be adjusted. This is actually something
        that should be more part of the manipulator viewer(s), but
        since that same value is shared by multiple such viewers
        it's easiest to attach it to the camera here ...*/
    float motionSpeed{1.f};
    float aspect{1.f};
    float fovyInDegrees{60.f};

    double lastModified = 0.;
};
} // namespace dtracker