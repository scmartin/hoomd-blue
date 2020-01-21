// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenh



#include "Manifold.h"
#include "hoomd/VectorMath.h"

namespace py = pybind11;

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

/*! \file Manifold.h
    \brief Declares a class that defines a differentiable manifold.
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \post The method is constructed with the given particle data and a NULL profiler.
*/
Manifold::Manifold(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group)
    : m_sysdef(sysdef), m_group(group), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf())
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    assert(m_group);
    }

void Manifold::setProfiler(std::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

//! Exports the Manifold class to python
void export_Manifold(pybind11::module& m)
    {
    py::class_< Manifold, std::shared_ptr<Manifold> >(m, "Manifold")
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >())
    .def("implicit_function", &Manifold::implicit_function)
    .def("derivative", &Manifold::derivative)
    ;
    }
