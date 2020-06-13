# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os
import numpy as np
import math

def almost_equal(u, v, e=0.001):
    return math.fabs(u-v)/math.fabs(u) <= e and math.fabs(u-v) / math.fabs(v) <= e;

# unit tests for md.integrate.brownian
class integrate_brownian_rattle_tests (unittest.TestCase):
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

    # tests basic creation of the integration method
    def test_basic_run(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=52);
        run(5);
        bd.disable();
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=1, dscale=1.0);
        run(5);
        bd.disable();
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=1, dscale=1.0, noiseless_t=True);
        run(5);
        bd.disable();
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=1, dscale=1.0, noiseless_r=True);
        run(5);
        bd.disable();
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=1, dscale=1.0, noiseless_t=True, noiseless_r=True);
        run(100);
        bd.disable();
        snapshot = self.system.take_snapshot()
        
        pos = snapshot.particles.position
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        for j in range(len(x)):
               self.assertAlmostEqual(x[j]**2+y[j]**2+z[j]**2,25,5)

    # test set_params
    def test_set_params(self):
        all = group.all();
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=1);
        bd.set_params(kT=1.3);

    # test set_gamma
    def test_set_gamma(self):
        all = group.all();
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=1);
        bd.set_gamma('A', 0.5);
        bd.set_gamma('B', 1.0);

    # test set_gamma
    def test_set_gamma_r(self):
        all = group.all();
        sphere = md.manifold.sphere(P=(0,0,0),r=5)
        bd = md.integrate.brownian_rattle(all, manifold=sphere, kT=1.2, seed=1);
        bd.set_gamma_r('A', 0.5);
        bd.set_gamma_r('B', (1.0,2.0,3.0));

    def tearDown(self):
        context.initialize();

class integrate_brownian_rattle_gyroid (unittest.TestCase):
    def setUp(self):
        print
        positions=np.array([[3.29052923,-7.88761357,5.99968317],
        [-2.91052384,-3.52035463,-7.9708974],
        [-8.78540484,0.58047014,-0.65154617],
        [5.57335844,2.42988619,9.50253932],
        [-7.20653425,-4.7672044,6.96489908],
        [9.62786868,-3.61985342,-2.24058276]])
        
        box = data.boxdim(L=20)
        snap = data.make_snapshot(N=positions.shape[0], box=box)
        if comm.get_rank() == 0:
            snap.particles.position[:] = positions
        self.system = init.read_snapshot(snap)

    # tests basic creation of the integration method
    def test_basic_run(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.001);
        gyroid = md.manifold.tpms(surface='Gyroid',N=1)
        bd = md.integrate.brownian_rattle(all, manifold=gyroid, kT=1.2, seed=52);
        run(5000);
        
        snap = self.system.take_snapshot();
        pos = snap.particles.position
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        for j in range(len(x)):
            self.assertAlmostEqual(np.sin(2*np.pi/20*x[j])*np.cos(2*np.pi/20*y[j]) +np.sin(2*np.pi/20*y[j])*np.cos(2*np.pi/20*z[j]) + np.sin(2*np.pi/20*z[j])*np.cos(2*np.pi/20*x[j]),0,5)

    def tearDown(self):
        context.initialize();


# validate brownian diffusion
class integrate_brownian_rattle_diffusion (unittest.TestCase):
    def setUp(self):
        print
        positions = np.empty((1000,3))
        positions[:]=(5,0,0)
        box = data.boxdim(L=20)
        snap = data.make_snapshot(N=positions.shape[0], box=box)
        if comm.get_rank() == 0:
            snap.particles.position[:] = positions
        self.s = init.read_snapshot(snap)

    def test_noiseless_t(self):
        # Setup an ideal gas with a gamma and T and validate the MSD
        kT=1.8
        gamma=1;
        dt=0.001;
        steps=5000;
        sphere = md.manifold.sphere(P=(0,0,0),r=5)

        md.integrate.mode_standard(dt=dt);
        bd = md.integrate.brownian_rattle(group.all(), manifold=sphere, kT=kT, seed=1, dscale=False, noiseless_t=True);
        bd.set_gamma('A', gamma);

        run(5000);

        snap = self.s.take_snapshot();
        if comm.get_rank() == 0:
            msd = np.mean((snap.particles.position[:,0]-5) * (snap.particles.position[:,0]-5) +
                             snap.particles.position[:,1] * snap.particles.position[:,1] +
                             snap.particles.position[:,2] * snap.particles.position[:,2])

            D = -np.log(1.0 - msd / (2*5*5) )*5.0*5.0/ (2*dt*steps);

            # check for a very crude overlap - we are not doing much averaging here to keep the test short
            self.assert_(math.fabs(D) < 0.1)

    def test_gamma(self):
        # Setup an ideal gas with a gamma and T and validate the MSD
        kT=1.8
        gamma=100;
        dt=0.01;
        steps=5000;
        sphere = md.manifold.sphere(P=(0,0,0),r=5)

        md.integrate.mode_standard(dt=dt);
        bd = md.integrate.brownian_rattle(group.all(), manifold=sphere, kT=kT, seed=1, dscale=False);
        bd.set_gamma('A', gamma);

        run(steps);

        snap = self.s.take_snapshot();
        if comm.get_rank() == 0:
            msd = np.mean((snap.particles.position[:,0]-5) * (snap.particles.position[:,0]-5) +
                             snap.particles.position[:,1] * snap.particles.position[:,1] +
                             snap.particles.position[:,2] * snap.particles.position[:,2])

            D = -np.log(1.0 - msd / (2*5*5) )*5.0*5.0/ (2*dt*steps);

            # check for a very crude overlap - we are not doing much averaging here to keep the test short
            self.assert_(almost_equal(D, kT/gamma, 0.1))

    def test_dscale(self):
        # Setup an ideal gas with a gamma and T and validate the MSD
        kT=1.8
        gamma=100;
        dt=0.01;
        steps=5000;
        sphere = md.manifold.sphere(P=(0,0,0),r=5)

        md.integrate.mode_standard(dt=dt);
        bd = md.integrate.brownian_rattle(group.all(), manifold=sphere, kT=kT, seed=1, dscale=gamma);

        run(steps);

        snap = self.s.take_snapshot();
        if comm.get_rank() == 0:
            msd = np.mean((snap.particles.position[:,0]-5) * (snap.particles.position[:,0]-5) +
                             snap.particles.position[:,1] * snap.particles.position[:,1] +
                             snap.particles.position[:,2] * snap.particles.position[:,2])

            D = -np.log(1.0 - msd / (2*5*5) )*5.0*5.0/ (2*dt*steps);

            # check for a very crude overlap - we are not doing much averaging here to keep the test short
            self.assert_(almost_equal(D, kT/gamma, 0.1))

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
