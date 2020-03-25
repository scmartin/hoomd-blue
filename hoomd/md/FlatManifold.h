// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "Manifold.h"

/*! \file FlatManifold.h
    \brief Declares the implicit function of a sphere.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __FLAT_MANIFOLD_H__
#define __FLAT_MANIFOLD_H__

//! Defines the geometry of a manifold.
class PYBIND11_EXPORT FlatManifold : public Manifold
    {
    public:
        //! Constructs the compute
        /*! \param surf Defines the specific plane
            \param shift Shift of the plane in normal direction.
        */
        FlatManifold(std::shared_ptr<SystemDefinition> sysdef,
                  std::string surf, 
                  Scalar shift);

        //! Destructor
        virtual ~FlatManifold();

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
        Scalar implicit_function(Scalar3 point);

        //! Return the gradient of the implicit function/normal vector.
        /*! \param point The location to evaluate the gradient.
        */
        Scalar3 derivative(Scalar3 point);

        Scalar returnLx();
    protected:
        std::string m_surf; //! determines which plane is considered
        Scalar m_shift; //! shift in normal direction

    private:
        //! Validate that the sphere is in the box and all particles are very near the constraint
	bool xy=false;
	bool xz=false;
	bool yz=false;
    };

//! Exports the FlatManifold class to python
void export_FlatManifold(pybind11::module& m);

#endif
