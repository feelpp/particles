#pragma once
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <optional>
#include <cmath>
#include <iostream>
#include <nelder_mead.hpp>

using Vec2 = Eigen::Vector2d;
using Vector = Eigen::VectorXd;

struct CSVLogger {
    mutable std::ofstream file;
    CSVLogger(const std::string& filename, const std::string& header) {
        file.open(filename);
        file << header << "\n";
    }

    template<typename... Args>
    void log(Args... args) const {
        bool first = true;
        ((file << (first ? "" : ",") << args, first = false), ...);
        file << "\n";
    }
};

struct Particle {
    Vec2 position;
    Vec2 velocity;
    Vec2 direction;
    double radius;
    double mass;
    double density;

    Particle(const Vec2& p, const Vec2& d, double r, double m)
        : position(p),
          velocity(Vec2::Zero()),
          direction(d.normalized()),
          radius(r),
          mass(m),
          density(m / (M_PI * r * r)) {}
};

struct Wall {
    Vec2 normal;
    double offset;
    Wall(Vec2 n, double d) : normal(n.normalized()), offset(d) {}

    double distance(const Vec2& p) const {
        return normal.dot(p) - offset;
    }

    Vec2 force(const Vec2& p, double radius, double k, double alpha, double dmin, double dcut) const {
        double d = distance(p) - radius;
        if (d >= dcut || d < 1e-6) return Vec2::Zero();
        double exponent = std::clamp(d - dmin, 1e-3, 1.0);
        double f = std::clamp(-k * std::pow(exponent, -alpha), -1e2, 1e2);
        return f * normal;
    }
};

std::vector<Particle> initialize_particles(int n, double box_size, std::mt19937& gen);

Vec2 compute_interparticle_force(const Particle& a, const Particle& b, double k, double alpha, double dmin, double dcut);

Vec2 compute_total_force(int i, const std::vector<Particle>& particles,
                         const std::vector<Wall>& walls,
                         double k, double alpha, double dmin, double dcut,
                         double propulsion_strength);

void update_particle(Particle& p, const Vec2& force, double dt, double damping);

double compute_step_penalty(const std::vector<Particle>& particles, double dmin, double& dmin_obs, int& n_collisions);

double simulate(const Vector& params,
                int n_particles,
                int steps,
                double box_size,
                const std::optional<CSVLogger>& trajectory_logger,
                const std::optional<CSVLogger>& metrics_logger);

double simulate_and_score(const Vector& params);
