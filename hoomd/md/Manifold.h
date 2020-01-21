// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "hoomd/VectorMath.h"

/*! \file Manifold.h
    \brief Declares a class that defines a differentiable manifold.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __MANIFOLD_H__
#define __MANIFOLD_H__

//! Defines the geometry of a manifold.
class PYBIND11_EXPORT Manifold
    {
    public:
        //! Constructs the compute. Does nothing in base class.
        Manifold() {};

        //! Destructor
        virtual ~Manifold() {}

        //! Return the value of the implicit surface function describing the manifold F(x,y,z)=0.
        /*! \param point The location to evaluate the implicit surface function.
        */
        virtual Scalar implicit_function(Scalar3 point) {return 0;}

        //! Return the derivative of the implicit function/normal vector
        /*! \param point The position to evaluate the derivative.
        */
        virtual Scalar3 derivative(Scalar3 point) {return make_scalar3(0, 0, 0);}

    };

//! Exports the Manifold class to python
inline void export_Manifold(pybind11::module& m)
    {
    pybind11::class_< Manifold, std::shared_ptr<Manifold> >(m, "Manifold")
    .def(pybind11::init<>())
    .def("implicit_function", &Manifold::implicit_function)
    .def("derivative", &Manifold::derivative)
    ;
    }

#endif
