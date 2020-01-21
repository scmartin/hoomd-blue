// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander



#include "TwoStepRATTLENVE.h"
#include "hoomd/VectorMath.h"


using namespace std;
namespace py = pybind11;

/*! \file TwoStepRATTLENVE.h
    \brief Contains code for the TwoStepRATTLENVE class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param manifold The manifold describing the constraint during the RATTLE integration method
    \param skip_restart Skip initialization of the restart information
*/
TwoStepRATTLENVE::TwoStepRATTLENVE(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       std::shared_ptr<Manifold> manifold,
                       bool skip_restart
                       Scalar eta)
    : IntegrationMethodTwoStep(sysdef, group), m_manifold(manifold), m_eta(eta), m_limit(false), m_limit_val(1.0), m_zero_force(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepRATTLENVE" << endl;

    if (!skip_restart)
        {
        // set a named, but otherwise blank set of integrator variables
        IntegratorVariables v = getIntegratorVariables();

        if (!restartInfoTestValid(v, "RATTLEnve", 0))
            {
            v.type = "RATTLEnve";
            v.variable.resize(0);
            setValidRestart(false);
            }
        else
            setValidRestart(true);

        setIntegratorVariables(v);
        }
    }

TwoStepRATTLENVE::~TwoStepRATTLENVE()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepRATTLENVE" << endl;
    }

/*! \param limit Distance to limit particle movement each time step

    Once the limit is set, future calls to update() will never move a particle
    a distance larger than the limit in a single time step
*/
void TwoStepRATTLENVE::setLimit(Scalar limit)
    {
    m_limit = true;
    m_limit_val = limit;
    }

/*! Disables the limit, allowing particles to move normally
*/
void TwoStepRATTLENVE::removeLimit()
    {
    m_limit = false;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepRATTLENVE::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("RATTLENVE step 1");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    unsigned int maxiteration = 10;


    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        if (m_zero_force)
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;

	Scalar lambda = 0.0;
        
	Scalar3 next_pos = Scalar3(h_pos.data[j]);
	Scalar3 normal = m_manifold->derivative(next_pos);

	Scalar inv_mass = Scalar(1.0)/h_vel.data[j].w;
	Scalar inv_alpha = -Scalar(1.0/2.0)*m_deltaT*m_deltaT*inv_mass;
	inv_alpha = Scalar(1.0)/inv_alpha;

	Scalar4 residual;
	
	unsigned int iteration = 0;
	do
	{
	    iteration++;
            vel_half.x = h_vel.data[j].x + Scalar(1.0)/Scalar(2.0)*m_deltaT*(h_accel.data[j].x-inv_mass*lambda*normal.x);
            vel_half.y = h_vel.data[j].y + Scalar(1.0)/Scalar(2.0)*m_deltaT*(h_accel.data[j].y-inv_mass*lambda*normal.y);
            vel_half.z = h_vel.data[j].z + Scalar(1.0)/Scalar(2.0)*m_deltaT*(h_accel.data[j].z-inv_mass*lambda*normal.z);

	    residual.x = h_pos.data[j].x - next_pos.x + m_deltaT*v_half.x;
	    residual.y = h_pos.data[j].y - next_pos.y + m_deltaT*v_half.y;
	    residual.z = h_pos.data[j].z - next_pos.z + m_deltaT*v_half.z;
	    residual.w = m_manifold->implicit_function(next_pos.x);

            Scalar3 next_normal =  m_manifold->derivative(next_pos);
	    Scalar beta = (residual.w + dot(next_normal,Scalar3(residual)))/dot(next_normal,normal);
	    
            next_pos.x -= (beta*normal.x - residual.x);   
            next_pos.y -= (beta*normal.y - residual.y);   
            next_pos.z -= (beta*normal.z - residual.z);   
	    lambda -= beta*inv_alpha;

	} while (maxNorm(residual) > eta && iteration < maxiteration );

        Scalar dx = next_pos.x - h_pos.data[j].x;
        Scalar dy = next_pos.y - h_pos.data[j].y;
        Scalar dz = next_pos.z - h_pos.data[j].z;

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar len = sqrt(dx*dx + dy*dy + dz*dz);
            if (len > m_limit_val)
                {
                dx = dx / len * m_limit_val;
                dy = dy / len * m_limit_val;
                dz = dz / len * m_limit_val;
                }
            }

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;

        h_vel.data[j].x += vel_half.x;
        h_vel.data[j].y += vel_half.y;
        h_vel.data[j].z += vel_half.z;
        }

    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        box.wrap(h_pos.data[j], h_image.data[j]);
        }

    // Integration of angular degrees of freedom using symplectic and
    // time-reversal symmetric integration scheme of Miller et al.
    if (m_aniso)
        {
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            quat<Scalar> q(h_orientation.data[j]);
            quat<Scalar> p(h_angmom.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
            vec3<Scalar> I(h_inertia.data[j]);

            // rotate torque into principal frame
            t = rotate(conj(q),t);

            // check for zero moment of inertia
            bool x_zero, y_zero, z_zero;
            x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

            // ignore torque component along an axis for which the moment of inertia zero
            if (x_zero) t.x = 0;
            if (y_zero) t.y = 0;
            if (z_zero) t.z = 0;

            // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
            // using Trotter factorization of rotation Liouvillian
            p += m_deltaT*q*t;

            quat<Scalar> p1, p2, p3; // permutated quaternions
            quat<Scalar> q1, q2, q3;
            Scalar phi1, cphi1, sphi1;
            Scalar phi2, cphi2, sphi2;
            Scalar phi3, cphi3, sphi3;

            if (!z_zero)
                {
                p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
                q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
                phi3 = Scalar(1./4.)/I.z*dot(p,q3);
                cphi3 = slow::cos(Scalar(1./2.)*m_deltaT*phi3);
                sphi3 = slow::sin(Scalar(1./2.)*m_deltaT*phi3);

                p=cphi3*p+sphi3*p3;
                q=cphi3*q+sphi3*q3;
                }

            if (!y_zero)
                {
                p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
                q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
                phi2 = Scalar(1./4.)/I.y*dot(p,q2);
                cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
                sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

                p=cphi2*p+sphi2*p2;
                q=cphi2*q+sphi2*q2;
                }

            if (!x_zero)
                {
                p1 = quat<Scalar>(-p.v.x,vec3<Scalar>(p.s,p.v.z,-p.v.y));
                q1 = quat<Scalar>(-q.v.x,vec3<Scalar>(q.s,q.v.z,-q.v.y));
                phi1 = Scalar(1./4.)/I.x*dot(p,q1);
                cphi1 = slow::cos(m_deltaT*phi1);
                sphi1 = slow::sin(m_deltaT*phi1);

                p=cphi1*p+sphi1*p1;
                q=cphi1*q+sphi1*q1;
                }

            if (! y_zero)
                {
                p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
                q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
                phi2 = Scalar(1./4.)/I.y*dot(p,q2);
                cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
                sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

                p=cphi2*p+sphi2*p2;
                q=cphi2*q+sphi2*q2;
                }

            if (! z_zero)
                {
                p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
                q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
                phi3 = Scalar(1./4.)/I.z*dot(p,q3);
                cphi3 = slow::cos(Scalar(1./2.)*m_deltaT*phi3);
                sphi3 = slow::sin(Scalar(1./2.)*m_deltaT*phi3);

                p=cphi3*p+sphi3*p3;
                q=cphi3*q+sphi3*q3;
                }

            // renormalize (improves stability)
            q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));

            h_orientation.data[j] = quat_to_scalar4(q);
            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepRATTLENVE::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("RATTLENVE step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        if (m_zero_force)
            {
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
            }
        else
            {
            // first, calculate acceleration from the net force
            Scalar minv = Scalar(1.0) / h_vel.data[j].w;
            h_accel.data[j].x = h_net_force.data[j].x*minv;
            h_accel.data[j].y = h_net_force.data[j].y*minv;
            h_accel.data[j].z = h_net_force.data[j].z*minv;
            }

           Scalar mu = 0;
           Scalar inv_alpha = -Scalar(1.0/2.0)*m_deltaT;
	   inv_alpha = Scalar(1.0)/inv_alpha;
   
           Scalar3 normal = m_manifold->derivative(Scalar3(h_pos.data[j]));
   
           Scalar3 next_vel = Scalar3(h_vel.data[j]) + Scalar(1.0/2.0)*m_deltaT*Scalar3(h_accel.data[j]);
	   Scalar mass = h_vel.data[j].w;
	   Scalar inv_mass = Scalar(1.0)/mass;
   
           Scalar4 residual;
           Scalar3 vel_dot;
   
           unsigned int iteration = 0;
           do
               {
               iteration++;
               v_dot.x = h_accel.data[j].x - mu*inv_mass*normal.x;
               v_dot.y = h_accel.data[j].y - mu*inv_mass*normal.y;
               v_dot.z = h_accel.data[j].z - mu*inv_mass*normal.z;

               resid.x = h_vel.data[j].x - next_v.x + Scalar(1.0/2.0)*m_deltaT*v_dot.x;
               resid.y = h_vel.data[j].y - next_v.y + Scalar(1.0/2.0)*m_deltaT*v_dot.y;
               resid.z = h_vel.data[j].z - next_v.z + Scalar(1.0/2.0)*m_deltaT*v_dot.z;
               resid.w = dot(normal, v_next);
   
               Scalar beta = (mass*resid.w + dot(normal, Scalar3(residual)))/dot(normal,normal);
               next_v.x -= (normal.x*beta-residual.x);
               next_v.y -= (normal.y*beta-residual.y);
               next_v.z -= (normal.z*beta-residual.z);
               mu -=  beat*inv_alpha;
   
	       } while (maxNorm(residual)*mass > eta && iteration < maxiteration );
   
           // then, update the velocity
           h_vel.data[j].x = v_next.x;
           h_vel.data[j].y = v_next.y;
           h_vel.data[j].z = v_next.z;

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar vel = sqrt(h_vel.data[j].x*h_vel.data[j].x+h_vel.data[j].y*h_vel.data[j].y+h_vel.data[j].z*h_vel.data[j].z);
            if ( (vel*m_deltaT) > m_limit_val)
                {
                h_vel.data[j].x = h_vel.data[j].x / vel * m_limit_val / m_deltaT;
                h_vel.data[j].y = h_vel.data[j].y / vel * m_limit_val / m_deltaT;
                h_vel.data[j].z = h_vel.data[j].z / vel * m_limit_val / m_deltaT;
                }
            }
        }

    if (m_aniso)
        {
        // angular degrees of freedom
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            quat<Scalar> q(h_orientation.data[j]);
            quat<Scalar> p(h_angmom.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
            vec3<Scalar> I(h_inertia.data[j]);

            // rotate torque into principal frame
            t = rotate(conj(q),t);

            // check for zero moment of inertia
            bool x_zero, y_zero, z_zero;
            x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

            // ignore torque component along an axis for which the moment of inertia zero
            if (x_zero) t.x = 0;
            if (y_zero) t.y = 0;
            if (z_zero) t.z = 0;

            // advance p(t+deltaT/2)->p(t+deltaT)
            p += m_deltaT*q*t;

            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepRATTLENVE(py::module& m)
    {
    py::class_<TwoStepRATTLENVE, std::shared_ptr<TwoStepRATTLENVE> >(m, "TwoStepRATTLENVE", py::base<IntegrationMethodTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<Manifold>, bool >())
        .def("setLimit", &TwoStepRATTLENVE::setLimit)
        .def("removeLimit", &TwoStepRATTLENVE::removeLimit)
        .def("setZeroForce", &TwoStepRATTLENVE::setZeroForce)
        ;
    }
