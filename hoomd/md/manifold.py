# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Manifold.

Manifold defining a positional constraint to a given set of particles. For example, a group of particles 
can be constrained to the surface of a sphere with :py:class:`sphere`.

Warning:
    Only one manifold can be applied to the integrators/active forces.

The degrees of freedom removed from the system by constraints are correctly taken into account when computing the
temperature for thermostatting and logging.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.manifold import _manifold
import hoomd;

class ellipsoid(_manifold):
    def __init__(self,a,b,c, P):
        hoomd.util.print_status_line();
        # initialize the base class
        _manifold.__init__(self);
        P = _hoomd.make_scalar3(P[0], P[1], P[2]);
        self.cpp_manifold = _md.EllipsoidManifold(hoomd.context.current.system_definition, a, b, c, P );

class plane(_manifold):
    def __init__(self,surface, shift):
        hoomd.util.print_status_line();
        # initialize the base class
        _manifold.__init__(self);
        surface = surface.upper();
        surface_list = ['XY','YX','XZ','ZX','YZ','ZY'];
        if surface not in surface_list:
            hoomd.context.msg.error("Specified Plane is not implemented\n");
            raise RuntimeError('Error creating manifold');

        if shift is None:
            shift = 0;
        self.cpp_manifold = _md.FlatManifold(hoomd.context.current.system_definition, surface, shift );

class sphere(_manifold):
    def __init__(self,r, P):
        hoomd.util.print_status_line();
        # initialize the base class
        _manifold.__init__(self);
        P = _hoomd.make_scalar3(P[0], P[1], P[2]);
        self.cpp_manifold = _md.SphereManifold(hoomd.context.current.system_definition, r, P );

class tpms(_manifold):
    def __init__(self,surface,N=None,Nx=None,Ny=None,Nz=None):
        hoomd.util.print_status_line();
        # initialize the base class
        _manifold.__init__(self);
        surface = surface.upper();
        surface_list = ['G','GYROID','D','DIAMOND','P','PRIMITIVE'];
        if surface not in surface_list:
            hoomd.context.msg.error("TPMS surface is not implemented\n");
            raise RuntimeError('Error creating manifold');
		
        if N is not None:
            Nx = Ny = Nz = N;

        self.cpp_manifold = _md.TPMSManifold(hoomd.context.current.system_definition, surface, Nx, Ny, Nz );

        # store metadata
        self.surface = surface
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.metadata_fields = ['surface','Nx','Ny','Nz']
