#ifndef BEZIER_LENGTH_H
#define BEZIER_LENGTH_H

#include <Eigen/Core>
#include <vector>

#include "bezier_curves.h"
#include "../biem/legendre_gauss_quadrature.h"
#include "../util/types.h"

namespace infinite{
    void bezier_length(const Mat42_list &CPs, 
                    const int &nl,
                    Eigen::VectorXd &L);

    void bezier_length(const Mat42_list &CPs, 
                    const int &ng,
                    const int &nl,
                    Eigen::VectorXd & wL);

    Eigen::VectorXd bezier_length(const Mat42_list &CPs);

    Eigen::VectorXd bezier_length(const Mat42_list &CPs, 
                    const int &ng);     

    Eigen::VectorXd cubic_length(const Mat42 &CP, 
                                const Eigen::MatrixX2d &TE, 
                                const int &ng);

    Eigen::VectorXd cubic_length(const Mat42 &CP, 
                                const Eigen::VectorXd &st,
                                const Eigen::VectorXd &et,
                                const int &ng);

    Eigen::VectorXd cubic_length(const Mat42 &CP, 
                                const Eigen::VectorXd &st,
                                const Eigen::VectorXd &et);
}      

#endif