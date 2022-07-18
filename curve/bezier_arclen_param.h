#ifndef BEZIER_ARCLEN_PARAM_H
#define BEZIER_ARCLEN_PARAM_H

#include <Eigen/Core>
#include <vector>

#include "bezier_curves.h"
#include "../biem/legendre_gauss_quadrature.h"
#include "../biem/interpolate_density.h"
#include "../util/types.h"

namespace infinite{
    // void bezier_arclen_param_old(const Mat42 &CP, 
    //                         const Eigen::VectorXd &T,
    //                         Eigen::VectorXd& TN);

    void bezier_arclen_param(const Mat42 &CP, 
                            const Eigen::VectorXd &T,
                            Eigen::VectorXd& TN);

    void bezier_arclen_param(const Mat42 &CP, 
                                  const double &t,
                                  double &tn);

}

#endif