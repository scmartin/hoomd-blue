// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __EVALUATOR_CONSTRAINT_Manifold_H__
#define __EVALUATOR_CONSTRAINT_Manifold_H__

#include "hoomd/HOOMDMath.h"
using namespace std;

/*! \file EvaluatorConstraintManifold.h
    \brief Defines the constraint evaluator class for ellipsoids
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating ellipsoid constraints
/*! <b>General Overview</b>
    EvaluatorConstraintManifold is a low level computation helper class to aid in evaluating particle constraints on a
    ellipsoid. Given a ellipsoid at a given position and radii, it will find the nearest point on the Manifold to a given
    position.
*/
class EvaluatorConstraintManifold
    {
    public:
        //! Constructs the constraint evaluator
        /*! \param _P Position of the ellipsoid
            \param _rx   Radius of the ellipsoid in the X direction
            \param _ry   Radius of the ellipsoid in the Y direction
            \param _rz   Radius of the ellipsoid in the Z direction

            NOTE: For the algorithm to work, we must have _rx >= _rz, ry >= _rz, and _rz > 0.
        */
        DEVICE EvaluatorConstraintManifold(Scalar3 _L, Scalar3 _R, unsigned int _surf)
            : L(_L), R(_R), surf(_surf)
            {
            }

        //! Evaluate the closest point on the ellipsoid. Method from: http://www.geometrictools.com/Documentation/DistancePointEllipseManifold.pdf
        /*! \param U unconstrained point

            \return Nearest point on the ellipsoid
        */

        DEVICE Scalar implicit_function(const Scalar3& U)
        {
            if( surf==1) return slow::sin(L.x*U.x)*slow::cos(L.y*U.y) + slow::sin(L.y*U.y)*slow::cos(L.z*U.z) + slow::sin(L.z*U.z)*slow::cos(L.x*U.x);	
	    else{ 
	    if(surf==2) return slow::cos(L.x*U.x)*slow::cos(L.y*U.y)*slow::cos(L.z*U.z) - slow::sin(L.x*U.x)*slow::sin(L.y*U.y)*slow::sin(L.z*U.z);
	    else{ 
            if(surf==3) return slow::cos(L.x*U.x) + slow::cos(L.y*U.y) + slow::cos(L.z*U.z);
            else{ 
            if(surf==4) return U.z;
            else{ 
            if(surf==5) return U.y;
            else{ 
            if(surf==6) return U.x;
            else{  
		 Scalar3 delta = U - L;
		return dot(delta, delta) - R.x*R.x;
	    }
            }
            }
            }
            }
            }
	}

        //! Evaluate the normal unit vector for point on the ellipsoid.
        /*! \param U point on ellipsoid
            \return normal unit vector for  point on the ellipsoid
        */
        DEVICE Scalar3 evalNormal(const Scalar3& U)
            {
            Scalar3 N= make_scalar3(0,0,0);
            if(surf == 1){
        	N.x = L.x*(slow::cos(L.x*U.x)*slow::cos(L.y*U.y) - slow::sin(L.z*U.z)*slow::sin(L.x*U.x));
          	N.y = L.y*(slow::cos(L.y*U.y)*slow::cos(L.z*U.z) - slow::sin(L.x*U.x)*slow::sin(L.y*U.y));
          	N.z = L.z*(slow::cos(L.z*U.z)*slow::cos(L.x*U.x) - slow::sin(L.y*U.y)*slow::sin(L.z*U.z)); 
	    }else{ 
	    if(surf==2){
                N.x = -L.x*(slow::sin(L.x*U.x)*slow::cos(L.y*U.y)*slow::cos(L.z*U.z) + slow::cos(L.x*U.x)*slow::sin(L.y*U.y)*slow::sin(L.z*U.z));
                N.y = -L.y*(slow::cos(L.x*U.x)*slow::sin(L.y*U.y)*slow::cos(L.z*U.z) + slow::sin(L.x*U.x)*slow::cos(L.y*U.y)*slow::sin(L.z*U.z));
                N.z = -L.z*(slow::cos(L.x*U.x)*slow::cos(L.y*U.y)*slow::sin(L.z*U.z) + slow::sin(L.x*U.x)*slow::sin(L.y*U.y)*slow::cos(L.z*U.z));
	    }else{
	    if(surf==3){
          	N.x = -L.x*slow::sin(L.x*U.x);
                N.y = -L.y*slow::sin(L.y*U.y);
                N.z = -L.z*slow::sin(L.z*U.z);
	    }else{
            if(surf==4) N.z = 1;
	    else{
            if(surf==5) N.y = 1;
	    else{
            if(surf==6) N.x = 1;
	    else N = 2*(U - L);
            }
            }
            }
            }
            }

            return N;
            }

    protected:
        Scalar3 L;      
        Scalar3 R;      
	unsigned int surf;
    };


#endif // __PAIR_EVALUATOR_LJ_H__
