#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>

typedef std::vector<Eigen::RowVector2d,Eigen::aligned_allocator<Eigen::RowVector2d> > RowVec2d_list;
typedef std::vector<RowVec2d_list> RowVec2d_list_list;

typedef std::vector<Eigen::Matrix2d,Eigen::aligned_allocator<Eigen::Matrix2d> > Mat2d_list;
typedef std::vector<Mat2d_list> Mat2d_list_list;

typedef Eigen::Matrix<double,4,2> Mat42;
typedef std::vector<Mat42,Eigen::aligned_allocator<Mat42> > Mat42_list;

typedef std::vector<Eigen::MatrixX3d,Eigen::aligned_allocator<Eigen::MatrixX3d> > MatX3d_list;

typedef std::vector<Eigen::MatrixXcd,Eigen::aligned_allocator<Eigen::MatrixXcd> > MatXcd_list;
typedef std::vector<Eigen::VectorXcd,Eigen::aligned_allocator<Eigen::VectorXcd> > VecXcd_list;

typedef std::vector<Eigen::RowVectorXd,Eigen::aligned_allocator<Eigen::RowVectorXd> > RowVecXd_list;

typedef std::vector<Eigen::VectorXd,Eigen::aligned_allocator<Eigen::VectorXd> > VecXd_list;
typedef std::vector<Eigen::VectorXi,Eigen::aligned_allocator<Eigen::VectorXi> > VecXi_list;

// row major eigen
typedef Eigen::Matrix<double,Eigen::Dynamic,2,Eigen::RowMajor> MatX2d;
typedef Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor> MatX3d;
typedef Eigen::Matrix<int,Eigen::Dynamic,2,Eigen::RowMajor> MatX2i;

#endif