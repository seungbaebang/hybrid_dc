#ifndef INTERPOLATE_DENSITY_H
#define INTERPOLATE_DENSITY_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

#include <igl/slice.h>
#include <igl/slice_into.h>

#include "legendre_transform.h"
#include "../util/types.h"

namespace infinite
{
    Eigen::MatrixXd interpolate_density(const Eigen::VectorXd& xg,
                                        const VecXd_list& xe_list,
                                        const Eigen::MatrixXd& sigma_g);

    Eigen::MatrixXd interpolate_density(const Eigen::VectorXd& xg,
                                        const Eigen::VectorXd& xe,
                                        const Eigen::MatrixXd& sigma_s);

    Eigen::VectorXd interpolate_density(const Eigen::VectorXd& xg,
                                        const Eigen::VectorXd& xe,
                                        const Eigen::VectorXd& sigma_s);

    double interpolate_density(const Eigen::VectorXd& xg,
                                const double& t,
                                const Eigen::VectorXd& sigma_s);  

    void update_density(const Eigen::MatrixXd &sigma,
                        const Eigen::VectorXi &sub_ids,
                        const Eigen::VectorXi &orig_ids,
                        const Eigen::VectorXi &new_sub_ids,
                        const Eigen::VectorXd &xg,
                        const double &sub_t,
                        Eigen::MatrixXd &sigma_n);

    void update_density(Eigen::MatrixXd &sigma,
                        const Eigen::VectorXi &sub_ids,
                        const Eigen::VectorXi &orig_ids,
                        const Eigen::VectorXi &new_sub_ids,
                        const Eigen::VectorXd &xg,
                        const double &sub_t);
}
#endif