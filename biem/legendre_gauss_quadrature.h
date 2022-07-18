#ifndef LEGENDRE_GAUSS_QUADRATURE_H
#define LEGENDRE_GAUSS_QUADRATURE_H

#include <Eigen/Core>
#include <vector>
#include <igl/PI.h>

namespace infinite{

    void legendre_gauss_quadrature(const int &ng, 
                const double &a, const double &b,
                Eigen::VectorXd &X, Eigen::VectorXd &W);

}

#endif