#include "simulation.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

//-----------------------------------------------------------------------------
// Global log file (opened once)
static std::ofstream log_file( "forces.log" );

//-----------------------------------------------------------------------------
// Particle initialization
std::vector<Particle> initialize_particles( int n, double box_size, std::mt19937& gen )
{
    std::vector<Particle> particles;
    std::uniform_real_distribution<> dist( 0.05, box_size - 0.05 );
    std::uniform_real_distribution<> angle_dist( 0.0, 2.0 * M_PI );
    std::uniform_real_distribution<> radius_dist( 0.01, 0.05 );
    std::uniform_real_distribution<> mass_dist( 0.05, 0.2 );
    while ( (int)particles.size() < n )
    {
        Vec2 pos( dist( gen ), dist( gen ) );
        Vec2 dir = Vec2( std::cos( angle_dist( gen ) ), std::sin( angle_dist( gen ) ) );
        double r = radius_dist( gen );
        bool ok = true;
        for ( auto& p : particles )
            if ( ( pos - p.position ).norm() < r + p.radius )
            {
                ok = false;
                break;
            }
        if ( !ok ) continue;
        double m = mass_dist( gen );
        particles.emplace_back( pos, dir, r, m );
        particles.back().velocity = dir * 0.2;
    }
    return particles;
}

//-----------------------------------------------------------------------------
// Spring-dashpot penalty magnitude: k*delta^2 + c*v_n
static double penalty_mag( double delta, double v_n, double k, double m_red, double restitution )
{
    double f_spring = k * delta * delta;
    double c = 2.0 * std::sqrt( k * m_red ) * ( 1.0 - restitution );
    double f_damp = c * v_n;
    return f_spring + f_damp;
}

//-----------------------------------------------------------------------------
// Repulsion between two particles
Vec2 compute_interparticle_force(
    const Particle& a,
    const Particle& b,
    double k,
    double alpha, // unused
    double dmin,  // unused
    double dcut )
{
    Vec2 rij = b.position - a.position;
    double d = rij.norm();
    double R = a.radius + b.radius;
    if ( d >= R + dcut ) return Vec2::Zero();
    Vec2 n = rij.normalized();
    double delta = ( R + dcut ) - d;
    Vec2 v_rel = b.velocity - a.velocity;
    double v_n = v_rel.dot( n );
    double inv_m1 = 1.0 / a.mass, inv_m2 = 1.0 / b.mass;
    double m_red = 1.0 / ( inv_m1 + inv_m2 );
    double Fmag = penalty_mag( delta, v_n, k, m_red, 0.8 );
    Vec2 F = -Fmag * n;
    log_file << "[pp] d=" << d << " delta=" << delta << " F=" << F.transpose() << "\n";
    return F;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Repulsion from wall using same spring-dashpot
Vec2 compute_wall_force(
    const Particle& p,
    const Wall& w,
    double k,
    double alpha, // unused
    double dmin,  // unused
    double dcut )
{
    double d = w.distance( p.position ) - p.radius;
    if ( d >= dcut ) return Vec2::Zero();
    Vec2 n = w.normal;
    // penetration into wall buffer
    double delta = dcut - d;
    // normal velocity component
    double v_n = p.velocity.dot( n );
    double m_red = p.mass;
    // penalty magnitude (spri  ng + damping)
    double Fmag = penalty_mag( delta, v_n, k, m_red, 0.8 );
    // outward repulsion direction: negate n
    Vec2 F = -Fmag * n;
    log_file << "[pw] d=" << d << " delta=" << delta << " F=" << F.transpose() << "\n";  
        return F;
}


//-----------------------------------------------------------------------------
// Compute total force on particle i
Vec2 compute_total_force(
    int i,
    const std::vector<Particle>& particles,
    const std::vector<Wall>& walls,
    double k,
    double alpha,
    double dmin,
    double dcut,
    double propulsion_strength )
{
    const Particle& pi = particles[i];
    Vec2 F = propulsion_strength * pi.direction;
    F -= 0.05 * pi.velocity; // drag
    for ( int j = 0; j < (int)particles.size(); ++j )
        if ( j != i )
            F += compute_interparticle_force( pi, particles[j], k, alpha, dmin, dcut );
    for ( auto& w : walls )
        F += compute_wall_force( pi, w, k, alpha, dmin, dcut );
    return F;
}

//-----------------------------------------------------------------------------
// Integration
void update_particle( Particle& p, const Vec2& F, double dt, double damping )
{
    Vec2 acc = F / p.mass;
    p.velocity += dt * acc;
    p.velocity *= damping;
    p.position += dt * p.velocity;
}

//-----------------------------------------------------------------------------
// Main simulation
double simulate(
    const Vector& params,
    int n_particles,
    int steps,
    double box_size,
    const std::optional<CSVLogger>& traj,
    const std::optional<CSVLogger>& metr )
{
    log_file.close();
    log_file.open( "forces.log", std::ios::trunc );
    double k = std::clamp( params[0], 1e-4, 1e3 );
    std::cout << "Spring constant k: " << k << "\n";
    double dcut = 0.05;
    double prop = 0.5;
    double dt = 0.005;
    double damp = 0.9;
    std::vector<Wall> walls = { Wall( { 1, 0 }, box_size ), Wall( { -1, 0 }, 0 ), Wall( { 0, 1 }, box_size ), Wall( { 0, -1 }, 0 ) };
    auto particles = initialize_particles( n_particles, box_size, *( new std::mt19937( 42 ) ) );
    for ( int t = 0; t < steps; ++t )
    {
        double time = t * dt;
        for ( int i = 0; i < n_particles; ++i )
        {
            Vec2 F = compute_total_force( i, particles, walls, k, 2, 0.01, dcut, prop );
            update_particle( particles[i], F, dt, damp );
            // wall clamping
            for ( int dim = 0; dim < 2; ++dim )
            {
                double& x = particles[i].position[dim];
                double& v = particles[i].velocity[dim];
                double r = particles[i].radius;
                if ( x < r )
                {
                    x = r;
                    if ( v < 0 ) v = -v * damp;
                }
                if ( x > box_size - r )
                {
                    x = box_size - r;
                    if ( v > 0 ) v = -v * damp;
                }
            }
            if ( traj ) traj->log( time, i, particles[i].position.x(), particles[i].position.y(), particles[i].radius );
        }
        if ( metr )
        {
            double dmin = 1e6;
            int nc = 0;
            double ek = 0;
            for ( int i = 0; i < n_particles; ++i )
            {
                ek += 0.5 * particles[i].velocity.squaredNorm();
                for ( int j = i + 1; j < n_particles; ++j )
                {
                    double d = ( particles[i].position - particles[j].position ).norm();
                    dmin = std::min( dmin, d );
                    if ( d < particles[i].radius + particles[j].radius ) ++nc;
                }
            }
            metr->log( time, dmin, nc, ek / n_particles );
        }
    }
    return 0;
}

double simulate_and_score( const Vector& params )
{
    return simulate( params, 25, 1000, 1.0, {}, {} );
}
