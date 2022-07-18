#ifndef DIFFCURV_COLOR_H
#define DIFFCURV_COLOR_H

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include "../util/types.h"

namespace infinite{

    Eigen::RowVector3d query_color(
                    const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const double &t);

    Eigen::RowVector3d eval_diffcurv_color(
                    const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const double &t);

    void eval_diffcurv_color(const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const Eigen::VectorXd &T,
                    Eigen::MatrixX3d &CR);

    void diffcurv_color(const MatX3d_list &Cs, //pre-defined color on curves
                        const VecXd_list &Xs, //pre-defined parametric value of color on curve
                        const VecXd_list &Ts, //parameter value to evaluate colors
                        Eigen::MatrixX3d &CR); //colors evaluated at Ts

    void diffcurv_color(const MatX3d_list &Cs, 
                        const VecXd_list &Ts,
                        const Eigen::VectorXd &T,
                        Eigen::MatrixX3d &CR);
}

#endif