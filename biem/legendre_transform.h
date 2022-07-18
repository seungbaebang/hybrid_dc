#ifndef LEGENDRE_TRANSFORM_H
#define LEGENDRE_TRANSFORM_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>

#include <igl/slice.h>

namespace infinite
{

    Eigen::RowVectorXd legendre_transform(const double &t, const int &n);

    Eigen::MatrixXd legendre_transform(const Eigen::VectorXd &T, const int &n);


    Eigen::MatrixXd legendre_interpolation(const Eigen::VectorXd &xg, 
                                        const Eigen::VectorXd &xc);

    void legendre_interpolation(const Eigen::VectorXd &xg, 
                        const Eigen::VectorXd &xc,
                        const int &nb,
                        Eigen::SparseMatrix<double> &ST);
}

#endif