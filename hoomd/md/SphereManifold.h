// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "hoomd/md/Manifold.h"

/*! \file SphereManifold.h
    \brief Declares the implicit function of a sphere.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __SPHERE_MANIFOLD_H__
#define __SPHERE_MANIFOLD_H__

//! Defines the geometry of a manifold.
class PYBIND11_EXPORT SphereManifold : public Manifold
    {
    public:
        //! Constructs the compute
        /*! \param r The r of the sphere.
            \param P The location of the sphere.
        */
        SphereManifold(Scalar r, Scalar3 P=make_scalar3(0,0,0));

        //! Destructor
        virtual ~SphereManifold();

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
        Scalar implicit_function(Scalar3 point);

        //! Return the gradient of the implicit function/normal vector.
        /*! \param point The location to evaluate the gradient.
        */
        Scalar3 derivative(Scalar3 point);

    private:
        Scalar m_r; //! The radius of the sphere.
        Scalar3 m_P; //! The center of the sphere.

    };

//! Exports the SphereManifold class to python
void export_SphereManifold(pybind11::module& m);

#endif
