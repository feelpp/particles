#pragma once
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>

struct NMResult
{
    Eigen::VectorXd x;
    double fx;
};

inline NMResult nelder_mead(
    std::function<double(const Eigen::VectorXd&)> f,
    Eigen::VectorXd const& start,
    double alpha = 1.0,
    double gamma = 2.0,
    double rho = 0.5,
    double sigma = 0.5,
    int max_iter = 200,
    double tol = 1e-6)
{
    const int n = start.size();
    std::vector<Eigen::VectorXd> simplex(n + 1, start);
    std::vector<double> fvals(n + 1);

    // Initialize simplex
    for (int i = 0; i < n; ++i)
    {
        Eigen::VectorXd point = start;
        point(i) += 0.05;
        simplex[i + 1] = point;
    }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        for (int i = 0; i <= n; ++i)
            fvals[i] = f(simplex[i]);

        std::vector<int> idx(n + 1);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int i, int j) { return fvals[i] < fvals[j]; });

        // Compute centroid
        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++i)
            x0 += simplex[idx[i]];
        x0 /= n;

        Eigen::VectorXd xr = x0 + alpha * (x0 - simplex[idx[n]]);
        double fr = f(xr);

        if (fr < fvals[idx[0]])
        {
            Eigen::VectorXd xe = x0 + gamma * (xr - x0);
            double fe = f(xe);
            simplex[idx[n]] = (fe < fr) ? xe : xr;
        }
        else if (fr < fvals[idx[n - 1]])
        {
            simplex[idx[n]] = xr;
        }
        else
        {
            Eigen::VectorXd xc = x0 + rho * (simplex[idx[n]] - x0);
            double fc = f(xc);
            if (fc < fvals[idx[n]])
            {
                simplex[idx[n]] = xc;
            }
            else
            {
                for (int i = 1; i <= n; ++i)
                    simplex[idx[i]] = simplex[idx[0]] + sigma * (simplex[idx[i]] - simplex[idx[0]]);
            }
        }

        // Convergence check
        double fmean = std::accumulate(fvals.begin(), fvals.end(), 0.0) / (n + 1);
        double var = 0.0;
        for (double v : fvals)
            var += std::pow(v - fmean, 2);
        var /= (n + 1);
        if (std::sqrt(var) < tol)
            break;
    }

    fvals[0] = f(simplex[0]);
    return {simplex[0], fvals[0]};
}