# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

import math

#---
# tests md.bond.harmonic
class constrain_distance_tests (unittest.TestCase):
    def setUp(self):
        print
        snap = data.make_snapshot(N=3,box=data.boxdim(L=25),particle_types=['A'])
        self.system = init.read_snapshot(snap)

        self.system.particles[0].position = (5,0,0)
        self.system.particles[1].position = (4,2,4)
        self.system.particles[2].position = (-0.82366867,-1.5844203,1.80553717)
    # test to see that se can create a md.force.constant
    def test_create(self):
        md.manifold.sphere(P=(0,0,0),r=5);
        md.manifold.ellipsoid(a=5,b=3,c=2,P=(0,0,0));
        md.manifold.tpms(surface='G',N=1);
        md.manifold.tpms(surface='diamond',N=2);
        md.manifold.tpms(surface='Primitive',N=3);

    def test_implicit(self):
        sphere1 = md.manifold.sphere(P=(0,0,0),r=5);
        sphere2 = md.manifold.sphere(P=(0,0,0),r=6);
        gyroid = md.manifold.tpms(surface='G',N=2);

        pos0 = self.system.particles[0].position
        pos1 = self.system.particles[1].position
        pos2 = self.system.particles[2].position
		
        self.assertAlmostEqual(sphere1.implicit_function(pos0),0,5)
        self.assertNotAlmostEqual(sphere1.implicit_function(pos1),0,5)
        self.assertNotAlmostEqual(sphere1.implicit_function(pos2),0,5)
        
        self.assertNotAlmostEqual(sphere2.implicit_function(pos0),0,5)
        self.assertAlmostEqual(sphere2.implicit_function(pos1),0,5)
        self.assertNotAlmostEqual(sphere2.implicit_function(pos2),0,5)
        
        self.assertNotAlmostEqual(gyroid.implicit_function(pos0),0,5)
        self.assertNotAlmostEqual(gyroid.implicit_function(pos1),0,5)
        self.assertAlmostEqual(gyroid.implicit_function(pos2),0,5)


    def test_derivative(self):
        sphere1 = md.manifold.sphere(P=(0,0,0),r=5);

        pos1 = self.system.particles[1].position
       	pos2 = sphere1.derivative(pos1)
        self.assertAlmostEqual(pos2.x,8,5)
        self.assertAlmostEqual(pos2.y,4,5)
        self.assertAlmostEqual(pos2.z,8,5)

    def tearDown(self):
        del self.system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
