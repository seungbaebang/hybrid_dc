#include "mat2d_to_compvec.h"

Eigen::VectorXcd mat2d_to_compvec(const Eigen::MatrixX2d &X)
{
  Eigen::VectorXcd V;
  V.resize(X.rows());
  V.real() = X.col(0);
  V.imag() = X.col(1);
  return V;
}