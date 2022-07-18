#ifndef SUBDIVIDE_CURVES_H
#define SUBDIVIDE_CURVES_H

#include <Eigen/Core>
#include <vector>
#include <iostream>

#include "diffcurv_color.h"
#include "bezier_length.h"
#include "bezier_arclen_param.h"
#include "../util/types.h"
#include "../util/edge_indices.h"

namespace infinite{
    void cubic_split(const Mat42 &CP,double t,Mat42 &CP1,Mat42 &CP2);

    void split_diffcurv_color(
                    const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const double &t,
                    Eigen::MatrixX3d &C1,
                    Eigen::VectorXd &X1,
                    Eigen::MatrixX3d &C2,
                    Eigen::VectorXd &X2);

    void subdivide_diffcurv(
                const Mat42_list &CPs,
                const MatX3d_list &CLCs,
                const VecXd_list &CLTs,
                const MatX3d_list &CRCs,
                const VecXd_list &CRTs,
                const double &t,
                Mat42_list &nCPs,
                MatX3d_list &nCLCs,
                VecXd_list &nCLTs,
                MatX3d_list &nCRCs,
                VecXd_list &nCRTs);

    void subdivide_diffcurv(
                Mat42_list &CPs,
                MatX3d_list &CLCs,
                VecXd_list &CLTs,
                MatX3d_list &CRCs,
                VecXd_list &CRTs,
                const double &t);

    Eigen::VectorXi get_adap_subdivision_list(
                const Eigen::VectorXd& L,
                const Eigen::VectorXd& xg,
                const Eigen::MatrixXd &sigma,
                const double &pixel_length);

}

#endif