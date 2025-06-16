#include "simulation.hpp"
#include <iostream>
#include <optional>

int main() {
    Vector start(1);
    start << 0.5; // initial epsilon

    CSVLogger optlog("optimization.csv", "iter,epsilon,cost");
    int iteration = 0;

    auto cost_fn = [&](const Vector& x) {
        Vector x_clamped = x;
        x_clamped[0] = std::clamp(x[0], 1e-4, 10.0); // epsilon bounds
        double cost = simulate_and_score(x_clamped);
        optlog.log(iteration++, x_clamped[0], cost);
        return cost;
    };

    auto result = nelder_mead(cost_fn, start);

    std::cout << "Optimal epsilon: " << result.x[0] << "\n";
    std::cout << "Final cost = " << result.fx << "\n";

    std::optional<CSVLogger> traj = CSVLogger("trajectories.csv", "time,id,x,y,radius");
    std::optional<CSVLogger> metr = CSVLogger("metrics.csv", "time,dmin,n_collisions,ekin");

    Vector result_clamped = result.x;
    result_clamped[0] = std::clamp(result.x[0], 1e-4, 10.0);

    simulate(result_clamped, 25, 1000, 1.0, traj, metr);

    return 0;
}