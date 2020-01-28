# -*- coding: iso-8859-1 -*-
# Maintainer: Pengji Zhou

import numpy as np
import os
import unittest
import math as m
from hoomd import *
from hoomd import md
context.initialize()

# md.pair.fourier
class pair_fourier_test (unittest.TestCase):
    def setUp(self):
        print
        snapshot = data.make_snapshot(N=2, box=data.boxdim(L=10))
        if context.current.device.comm.rank == 0:
            # suppose spherical particles
            snapshot.particles.position[0] = [0.0, 0.0, 0.0]
            snapshot.particles.position[1] = [0.5, 0.75, 1.0]
        self.system = init.read_snapshot(snapshot)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    def python_fourier_calculation(self, dx, dy, dz, fourier_a, fourier_b, degree=21, r_cut=3.0):
        # do fourier potential calculation by python
        r = np.sqrt(dx**2+dy**2+dz**2)
        n_shift = 2
        energy = 0
        force = 0
        if r > r_cut:
           force_x = 0
           force_y = 0
           force_z = 0
        else:
           a1 = 0
           for i in range(0, degree-1):
               a1 += fourier_a[i] * (-1)**i
           energy += a1 * np.cos(np.pi*r/r_cut)
           force += a1 * np.sin(np.pi*r/r_cut)*np.pi/r_cut

           for i in range(0, degree-1):
               energy += fourier_a[i] * np.cos(np.pi*r/r_cut*(i+n_shift))
               energy += fourier_b[i] * np.sin(np.pi*r/r_cut*(i+n_shift))
               force += fourier_a[i] * np.sin(np.pi*r/r_cut*(i+n_shift)) * np.pi/r_cut*(i+n_shift)
               force -= fourier_b[i] * np.cos(np.pi*r/r_cut*(i+n_shift)) * np.pi/r_cut*(i+n_shift)
           force += 2/r * energy
           force = force/r**2 + 12/r**13
           energy = energy/r**2 + 1/r**12
           force_x = force*dx/r
           force_y = force*dy/r
           force_z = force*dz/r
        return force_x, force_y, force_z, energy/2

    def test_pair_fourier_value(self):
        degree = 21
        r_cut = 3.0
        fourier = md.pair.fourier(r_cut=r_cut, nlist=self.nl)
        fourier_a = list((np.random.random(degree-1)-0.5)*2/degree)
        fourier_b = list((np.random.random(degree-1)-0.5)*2/degree)
        fourier.pair_coeff.set('A', 'A', fourier_a=fourier_a, fourier_b=fourier_b, degree=degree)
        md.integrate.mode_standard(dt=0.0)
        md.integrate.nve(group=group.all())
        run(1)
        dx = 0.5
        dy = 0.75
        dz = 1.0
        force_x, force_y, force_z, energy = self.python_fourier_calculation(dx, dy, dz, fourier_a, fourier_b, degree, r_cut)

        force_fourier_1 = self.system.particles[0].net_force
        potential_fourier_1 = self.system.particles[0].net_energy
        np.testing.assert_allclose(-force_x, force_fourier_1[0], rtol=1e-8)
        np.testing.assert_allclose(-force_y, force_fourier_1[1], rtol=1e-8)
        np.testing.assert_allclose(-force_z, force_fourier_1[2], rtol=1e-8)
        np.testing.assert_allclose(energy, potential_fourier_1, rtol=1e-8)

        force_fourier_2 = self.system.particles[1].net_force
        potential_fourier_2 = self.system.particles[1].net_energy
        np.testing.assert_allclose(force_x, force_fourier_2[0], rtol=1e-8)
        np.testing.assert_allclose(force_y, force_fourier_2[1], rtol=1e-8)
        np.testing.assert_allclose(force_z, force_fourier_2[2], rtol=1e-8)
        np.testing.assert_allclose(energy, potential_fourier_2, rtol=1e-8)

    def tearDown(self):
        del self.system, self.nl
        context.initialize()


if __name__ == '__main__':
    unittest.main(argv=['test.py', '-v'])
