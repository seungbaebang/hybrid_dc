#ifndef GREEN_QUADRATURE_H
#define GREEN_QUADRATURE_H

#include <Eigen/Core>
#include <vector>

#include <igl/PI.h>

namespace infinite
{

    void green_quadrature(const Eigen::VectorXcd &Pi,
                        const Eigen::VectorXcd &Ni,
                        const Eigen::VectorXd  &L,
                        const std::vector<int> &ids,
                        const std::complex<double> &qi,
                        Eigen::RowVectorXd &G,
                        Eigen::RowVectorXd &F);


    void green_quadrature(const Eigen::VectorXcd &Pi,
                        const Eigen::VectorXcd &Ni,
                        const Eigen::VectorXd  &L,
                        const Eigen::VectorXcd &Qi,
                        Eigen::MatrixXd &G,
                        Eigen::MatrixXd &F);

}

#endif