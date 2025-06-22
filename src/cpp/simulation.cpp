#include "simulation.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

//-----------------------------------------------------------------------------
// Global log file (opened once)
static std::ofstream log_file( "forces.log" );

//-----------------------------------------------------------------------------
std::vector<Particle> initialize_particles( int n, double box_size, std::mt19937& gen )
{
    std::vector<Particle> particles;
    std::uniform_real_distribution<> dist( 0.05, box_size - 0.05 );
    std::uniform_real_distribution<> angle_dist( 0.0, 2.0 * M_PI );
    std::uniform_real_distribution<> radius_dist( 0.01, 0.05 );
    std::uniform_real_distribution<> mass_dist( 0.05, 0.2 );
    double initial_speed = 0.2;

    int attempts = 0;
    while ( particles.size() < n && attempts < 5000 )
    {
        double theta = angle_dist( gen );
        Vec2 dir( std::cos( theta ), std::sin( theta ) );
        Vec2 pos( dist( gen ), dist( gen ) );
        double r = radius_dist( gen );
        double m = mass_dist( gen );

        bool overlap = false;
        for ( const auto& existing : particles )
        {
            if ( ( pos - existing.position ).norm() < r + existing.radius )
            {
                overlap = true;
                break;
            }
        }
        if ( !overlap )
        {
            particles.emplace_back( pos, dir, r, m );
            particles.back().velocity = dir * initial_speed;
        }
        ++attempts;
    }
    if ( particles.size() < n )
    {
        std::cerr << "⚠️ Could not place all particles without overlap ("
                  << particles.size() << "/" << n << ")\n";
    }
    return particles;
}

//-----------------------------------------------------------------------------
// Pairwise collision resolution with restitution
void resolve_collisions( std::vector<Particle>& particles, double restitution )
{
    const int N = particles.size();
    for ( int i = 0; i < N; ++i )
    {
        for ( int j = i + 1; j < N; ++j )
        {
            Particle &a = particles[i], &b = particles[j];
            Vec2 rij = b.position - a.position;
            double d = rij.norm();
            double R = a.radius + b.radius;
            if ( d <= 0 || d > R ) continue;

            // normalize
            Vec2 n = rij / d;
            double penetration = R - d;
            // positional correction: move half the penetration each
            a.position -= 0.5 * penetration * n;
            b.position += 0.5 * penetration * n;

            // relative velocity
            Vec2 v_rel = b.velocity - a.velocity;
            double v_n = v_rel.dot( n );
            if ( v_n > 0 ) continue; // separating

            // impulse scalar
            double inv_m1 = 1.0 / a.mass;
            double inv_m2 = 1.0 / b.mass;
            double J = -( 1 + restitution ) * v_n / ( inv_m1 + inv_m2 );
            Vec2 impulse = J * n;

            // apply impulse
            a.velocity -= impulse * inv_m1;
            b.velocity += impulse * inv_m2;

            log_file << "[coll] i=" << i << " j=" << j
                     << " pen=" << penetration
                     << " J=" << J
                     << " v_rel_n=" << v_n << "\n";
        }
    }
}

//-----------------------------------------------------------------------------
// Total force from self-propulsion, drag, and walls (no interparticle penalty)
Vec2 compute_total_force(
    const Particle& pi,
    const std::vector<Wall>& walls,
    double propulsion_strength,
    double drag_coef )
{
    Vec2 F = propulsion_strength * pi.direction;
    Vec2 drag = -drag_coef * pi.velocity;
    F += drag;
    for ( const auto& wall : walls )
    {
        F += wall.force( pi.position, pi.radius, 1e-3, 0.0, 0.0, 0.1 );
    }
    return F;
}

//-----------------------------------------------------------------------------
void update_particle( Particle& p, const Vec2& F, double dt )
{
    // integrate
    p.velocity += dt * ( F / p.mass );
    p.position += dt * p.velocity;
}

//-----------------------------------------------------------------------------
double simulate(
    const Vector& params,
    int n_particles,
    int steps,
    double box_size,
    const std::optional<CSVLogger>& trajectory_logger,
    const std::optional<CSVLogger>& metrics_logger )
{
    log_file.close();
    log_file.open( "forces.log", std::ios::trunc );
    log_file << "== Simulation start ==\n";

    double restitution = 0.8;
    double propulsion = params[0]; // tuneable
    double drag_coef = 0.05;
    double dt = 0.005;

    std::vector<Wall> walls = {
        Wall( { 1, 0 }, 1.0 ), Wall( { -1, 0 }, 0.0 ),
        Wall( { 0, 1 }, 1.0 ), Wall( { 0, -1 }, 0.0 ) };

    std::mt19937 gen( 42 );
    auto particles = initialize_particles( n_particles, box_size, gen );

    for ( int t = 0; t < steps; ++t )
    {
        // step forces
        for ( int i = 0; i < particles.size(); ++i )
        {
            Vec2 F = compute_total_force( particles[i], walls, propulsion, drag_coef );
            update_particle( particles[i], F, dt );
            // wall bounds with radius offset
            for ( int dim = 0; dim < 2; ++dim )
            {
                double& x = particles[i].position[dim];
                double& v = particles[i].velocity[dim];
                double r = particles[i].radius;
                if ( x < r )
                {
                    x = r;
                    if ( v < 0 ) v = -v * restitution;
                }
                else if ( x > 1 - r )
                {
                    x = 1 - r;
                    if ( v > 0 ) v = -v * restitution;
                }
            }
            if ( trajectory_logger )
                trajectory_logger->log( t * dt, i,
                                        particles[i].position.x(), particles[i].position.y(), particles[i].radius );
        }
        // resolve particle-particle collisions
        resolve_collisions( particles, restitution );
        // log kinetic
        if ( metrics_logger )
        {
            double ekin = 0;
            for ( auto& p : particles )
                ekin += 0.5 * p.velocity.squaredNorm();
            metrics_logger->log( t * dt, 0.0, 0, ekin / n_particles );
        }
    }
    return 0.0; // no penalty metric
}

//-----------------------------------------------------------------------------
double simulate_and_score( const Vector& params )
{
    return simulate( params, 25, 1000, 1.0, std::nullopt, std::nullopt );
}
