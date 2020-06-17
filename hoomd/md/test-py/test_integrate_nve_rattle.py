# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os
import numpy as np

# unit tests for md.integrate.nve
class integrate_nve_rattle_tests (unittest.TestCase):
    def setUp(self):
        print
        theta = np.linspace(0, np.pi, 10)
        phi = np.linspace(0, 2*np.pi, 10)
        theta, phi = np.meshgrid(theta, phi)
        points = np.stack((theta, phi), axis=2).reshape(-1, 2)
        theta = points[:, 0]
        phi = points[:, 1]
        sin_t = np.sin(theta)
        x = 5*sin_t*np.cos(phi)
        y = 5*sin_t*np.sin(phi)
        z = 5*np.cos(theta)
        positions = np.unique(np.round(np.stack((x, y, z), axis=1), 3), axis=0)
        box = data.boxdim(L=20)
        snap = data.make_snapshot(N=positions.shape[0], box=box)
        if comm.get_rank() == 0:
            snap.particles.position[:] = positions
        self.system = init.read_snapshot(snap)

    # tets basic creation of the dump
    def test_basic_run(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        md.integrate.nve_rattle(group=all,manifold=sphere);
        run(1000);

        snapshot = self.system.take_snapshot()
        
        pos = snapshot.particles.position
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        for j in range(len(x)):
               self.assertAlmostEqual(x[j]**2+y[j]**2+z[j]**2,25,5)
	
        

     #tests creation of the method with options
    def test_options(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        md.integrate.nve_rattle(group=all, manifold=sphere, limit=0.01, zero_force=True);
        run(100);

    # test set_params
    def test_set_params(self):
        all = group.all();
        mode = md.integrate.mode_standard(dt=0.005);
        mode.set_params(dt=0.001);
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        nve = md.integrate.nve_rattle(group=all,manifold=sphere);
        nve.set_params(limit=False);
        nve.set_params(limit=0.1);
        nve.set_params(zero_force=False);

    # test w/ empty group
    def test_empty(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = md.integrate.mode_standard(dt=0.005);
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        nve = md.integrate.nve_rattle(group=empty,manifold=sphere)
        run(1);

    # test method can be enabled and disabled
    def test_disable_enable(self):
        mode = md.integrate.mode_standard(dt=0.005);
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        nve = md.integrate.nve_rattle(group=group.all(),manifold=sphere)
        self.assertTrue(nve in context.current.integration_methods)

        # disable this integrator, which removes it from the context
        nve.disable()
        self.assertFalse(nve in context.current.integration_methods)
        # second call does nothing
        nve.disable()

        # reenable the integrator
        nve.enable()
        self.assertTrue(nve in context.current.integration_methods)
        # second call does nothing
        nve.enable()

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
