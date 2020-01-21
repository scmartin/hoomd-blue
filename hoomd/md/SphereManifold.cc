// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "SphereManifold.h"

namespace py = pybind11;

using namespace std;

/*! \file SphereManifold.cc
    \brief Contains code for the SphereManifold class
*/

/*!
    \param P position of the sphere
    \param r radius of the sphere
*/
SphereManifold::SphereManifold(Scalar r, Scalar3 P)
            : m_r(r), m_P(P) 
       {
       }

SphereManifold::~SphereManifold() 
       {
       }

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
Scalar SphereManifold::implicit_function(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       return dot(delta, delta) - m_r*m_r;
       }

       //! Return the gradient of the constraint.
       /*! \param point The location to evaluate the gradient.
       */
Scalar3 SphereManifold::derivative(Scalar3 point)
       {
       return point - m_P;
       }

//! Exports the SphereManifold class to python
void export_SphereManifold(pybind11::module& m)
    {
    pybind11::class_< SphereManifold, std::shared_ptr<SphereManifold> >(m, "SphereManifold", pybind11::base<Manifold>())
    .def(py::init<Scalar, Scalar3 >())
    .def("implicit_function", &SphereManifold::implicit_function)
    .def("derivative", &SphereManifold::derivative)
    ;
    }
