#ifndef IK_H
#define IK_H

#include <Eigen/Core>
#include <complex>

namespace infinite{

    std::complex<double> compute_Ik(const std::complex<double> &z, const int &k);
    std::complex<double> compute_Ok(const std::complex<double> &z, const int &k);

    void compute_Ik(const std::complex<double> &z, const int &np, Eigen::VectorXcd& Ik);

    void compute_Ik(const std::complex<double> &z, const int &np, Eigen::MatrixXcd& Ik, const int &j);

    void compute_Ok(const std::complex<double> &z, const int &k, std::complex<double> &v);

    void compute_Ok(const std::complex<double> &z, const int &np, Eigen::VectorXcd& Ok);

    void compute_Ok_no_log(const std::complex<double> &z, const int &np, Eigen::VectorXcd& Ok);

    void compute_Ok(const std::complex<double> &z, const int &np, Eigen::MatrixXcd& Ok, const int &j);

}

#endif