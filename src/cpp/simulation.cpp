#include "simulation.hpp"

std::vector<Particle> initialize_particles(int n, double box_size, std::mt19937& gen) {
    std::vector<Particle> particles;
    std::uniform_real_distribution<> dist(0.05, box_size - 0.05);
    std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<> radius_dist(0.01, 0.05);
    std::uniform_real_distribution<> mass_dist(0.1, 10.0); // lower mass to increase responsiveness

    for (int i = 0; i < n; ++i) {
        double theta = angle_dist(gen);
        Vec2 dir(std::cos(theta), std::sin(theta));
        Vec2 pos(dist(gen), dist(gen));
        double r = radius_dist(gen);
        double m = mass_dist(gen);
        particles.emplace_back(pos, dir, r, m);
    }
    return particles;
}

Vec2 compute_interparticle_force(const Particle& a, const Particle& b, double epsilon, double alpha_unused, double dmin_unused, double dcut_unused) {
    Vec2 rij = a.position - b.position;
    double d = rij.norm();
    double rhoL = 2.0 * std::max(a.radius, b.radius);

    if (d >= rhoL) return Vec2::Zero();

    double factor = (rhoL - d);
    double f = (1.0 / epsilon) * factor * factor;

    return f * rij.normalized();
}

Vec2 compute_total_force(int i, const std::vector<Particle>& particles,
                         const std::vector<Wall>& walls,
                         double epsilon, double alpha_unused, double dmin_unused, double dcut_unused,
                         double propulsion_strength) {
    const Particle& pi = particles[i];
    Vec2 F = (propulsion_strength / pi.radius) * pi.direction;

    for (int j = 0; j < particles.size(); ++j) {
        if (i == j) continue;
        F += compute_interparticle_force(pi, particles[j], epsilon, alpha_unused, dmin_unused, dcut_unused);
    }

    for (const auto& wall : walls) {
        F += wall.force(pi.position, pi.radius, epsilon, 2.0, 0.01, 0.1);
    }

    Vec2 drag = -0.1 * pi.velocity;
    F += drag;

    return F;
}

void update_particle(Particle& p, const Vec2& force, double dt, double damping) {
    Vec2 acc = force / p.mass;
    if (acc.norm() > 100)
        std::cerr << "\U0001F4A5 Accélération extrême: " << acc.norm() << "\n";
    p.velocity += dt * acc;
    p.velocity *= damping;
    Vec2 dp = dt * p.velocity;
    if (dp.norm() > 0.5 * p.radius) {
        std::cerr << "⚠️ Déplacement trop grand pour le rayon: " << dp.norm() << " > " << 0.5 * p.radius << "\n";
    }
    p.position += dp;

    for (int dim = 0; dim < 2; ++dim) {
        if (p.position[dim] < 0.0) {
            p.position[dim] = 0.0;
            p.velocity[dim] *= -0.8;
        }
        else if (p.position[dim] > 1.0) {
            p.position[dim] = 1.0;
            p.velocity[dim] *= -0.8;
        }
    }
}

double compute_step_penalty(const std::vector<Particle>& particles, double dmin, double& dmin_obs, int& n_collisions) {
    dmin_obs = 1e6;
    n_collisions = 0;
    double total = 0.0;
    double threshold = dmin + 0.005;
    for (int i = 0; i < particles.size(); ++i) {
        for (int j = i + 1; j < particles.size(); ++j) {
            double d = (particles[i].position - particles[j].position).norm();
            double min_dist = particles[i].radius + particles[j].radius;
            double gap = d - min_dist;
            dmin_obs = std::min(dmin_obs, d);
            if (gap < threshold)
                n_collisions++;
        }
    }
    return std::min(n_collisions * 10.0, 1e4);
}

double simulate(const Vector& params,
                int n_particles,
                int steps,
                double box_size,
                const std::optional<CSVLogger>& trajectory_logger,
                const std::optional<CSVLogger>& metrics_logger) {

    double epsilon = std::clamp(params[0], 1e-3, 10.0);
    double alpha_unused = 2.0;
    double dmin_unused = 0.01;
    double dcut_unused = 0.1;
    double dt    = 0.0005;
    double propulsion_strength = 100.0; // stronger propulsion
    double damping = 0.90;

    std::vector<Wall> walls = {
        Wall({1, 0}, 1.0),
        Wall({-1, 0}, 0.0),
        Wall({0, 1}, 1.0),
        Wall({0, -1}, 0.0)
    };

    std::mt19937 gen(42);
    auto particles = initialize_particles(n_particles, box_size, gen);

    double total_penalty = 0.0;

    for (int t = 0; t < steps; ++t) {
        for (int i = 0; i < particles.size(); ++i) {
            Vec2 F = compute_total_force(i, particles, walls, epsilon, alpha_unused, dmin_unused, dcut_unused, propulsion_strength);
            update_particle(particles[i], F, dt, damping);
            if (!particles[i].position.allFinite()) {
                std::cerr << "🚨 Particle " << i << " diverged at step " << t << "\n";
                return 1e10;
            }
            if (trajectory_logger)
                trajectory_logger->log(t * dt, i, particles[i].position.x(), particles[i].position.y(), particles[i].radius);
        }

        if (t % 100 == 0) {
            double avg_v = 0;
            for (const auto& p : particles) avg_v += p.velocity.norm();
            std::cerr << "Step " << t << " avg |v| = " << avg_v / particles.size() << "\n";
        }

        double dmin_obs;
        int n_collisions;
        double ekin = 0.0;
        for (const auto& p : particles)
            ekin += 0.5 * p.velocity.squaredNorm();

        if (!std::isfinite(ekin))
            return 1e10;

        total_penalty += compute_step_penalty(particles, dmin_unused, dmin_obs, n_collisions);

        if (metrics_logger)
            metrics_logger->log(t * dt, dmin_obs, n_collisions, ekin / n_particles);

        if (n_collisions > 1e6) {
            std::cerr << "⚠️ Too many collisions — early exit.\n";
            return 1e10;
        }
    }

    return total_penalty;
}

double simulate_and_score(const Vector& params) {
    return simulate(params, 25, 1000, 1.0, std::nullopt, std::nullopt);
}