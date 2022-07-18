#ifndef MAT2D_TO_COMPVEC_H
#define MAT2D_TO_COMPVEC_H

#include <Eigen/Core>

Eigen::VectorXcd mat2d_to_compvec(const Eigen::MatrixX2d &X);

#endif