#define BOOST_TEST_MODULE NelderMeadTest
#include <boost/test/included/unit_test.hpp>
#include <nelder_mead.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

BOOST_AUTO_TEST_CASE(minimum_of_parabola)
{
    auto f = [](const VectorXd& x) {
        return std::pow(x[0] - 3.0, 2);
    };

    VectorXd start(1);
    start << 0.0;

    auto result = nelder_mead(f, start);

    BOOST_CHECK_SMALL(result.x[0] - 3.0, 1e-3);
    BOOST_CHECK_SMALL(result.fx, 1e-6);
}

BOOST_AUTO_TEST_CASE(minimum_of_2d_quadratic)
{
    auto f = [](const VectorXd& x) {
        return std::pow(x[0] - 1.0, 2) + std::pow(x[1] + 2.0, 2);
    };

    VectorXd start(2);
    start << 0.0, 0.0;

    auto result = nelder_mead(f, start);

    BOOST_CHECK_SMALL(result.x[0] - 1.0, 1e-3);
    BOOST_CHECK_SMALL(result.x[1] + 2.0, 1e-3);
    BOOST_CHECK_SMALL(result.fx, 1e-6);
}