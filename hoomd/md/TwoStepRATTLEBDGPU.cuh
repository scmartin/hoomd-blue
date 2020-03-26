// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TwoStepRATTLEBDGPU.cuh
    \brief Declares GPU kernel code for Brownian dynamics on the GPU. Used by TwoStepRATTLEBDGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "TwoStepRATTLELangevinGPU.cuh"
#include "EvaluatorConstraintManifold.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUPartition.cuh"

#ifndef __TWO_STEP_RATTLE_BD_GPU_CUH__
#define __TWO_STEP_RATTLE_BD_GPU_CUH__

//! Kernel driver for the first part of the Brownian update called by TwoStepRATTLEBDGPU
cudaError_t gpu_rattle_brownian_step_one(Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  int3 *d_image,
                                  const BoxDim &box,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  const unsigned int *d_group_members,
                                  const unsigned int group_size,
                                  const Scalar4 *d_net_force,
                                  const Scalar3 *d_gamma_r,
                                  Scalar4 *d_orientation,
                                  Scalar4 *d_torque,
                                  const Scalar3 *d_inertia,
                                  Scalar4 *d_angmom,
                                  const rattle_langevin_step_two_args& rattle_langevin_args,
				  EvaluatorConstraintManifold manifold,
                                  const bool aniso,
                                  const Scalar deltaT,
                                  const unsigned int D,
                                  const bool d_noiseless_t,
                                  const bool d_noiseless_r,
                                  const GPUPartition& gpu_partition);

#endif //__TWO_STEP_RATTLE_BD_GPU_CUH__
