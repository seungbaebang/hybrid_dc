#ifndef EDGE_INDICES_H
#define EDGE_INDICES_H

#include <Eigen/Core>
#include <vector>

typedef Eigen::Matrix<int,Eigen::Dynamic,2,Eigen::RowMajor> MatX2i;

Eigen::MatrixX2i edge_indices(int nb, int ne);

void edge_indices(int nb, int ne, MatX2i &E);

void edge_indices(const Eigen::VectorXi &nel, Eigen::MatrixX2i &E);

// template <class Mat>
// Mat edge_indices(int nb, int ne);


#endif
