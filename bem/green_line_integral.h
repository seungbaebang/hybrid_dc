#ifndef GREEN_LINE_INTEGRAL_H
#define GREEN_LINE_INTEGRAL_H

#include <Eigen/Core>
#include <vector>
#include <igl/PI.h>
#include "../util/types.h"


namespace infinite
{

    double green_line_integral(const Eigen::Matrix2d &PEs,
                         const Eigen::RowVector2d &Q);

    double green_line_integral(const Eigen::Matrix2d &PEs,
                         const double  &l,
                         const Eigen::RowVector2d &Q);

    void green_line_integral(const Mat2d_list &PEs,
                         const Eigen::VectorXd &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G);

    void green_line_integral(const Mat2d_list &PEs,
                         const Eigen::MatrixX2d &N,
                         const Eigen::VectorXd &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G,
                         Eigen::RowVectorXd &F);

    void green_line_integral_test(const Mat2d_list &PEs,
                         const Eigen::MatrixX2d &N,
                         const std::vector<double>  &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G,
                         Eigen::RowVectorXd &Gf,
                         Eigen::RowVectorXd &F);

    void green_line_integral(const Mat2d_list &PEs,
                            const Eigen::MatrixX2d &N,
                            const std::vector<double>  &Ls,
                            const Eigen::RowVector2d &Q,
                            Eigen::RowVectorXd &G,
                            Eigen::RowVectorXd &F);

    void green_line_integral(const Mat2d_list &PEs,
                            const std::vector<double> &Ls,
                            const Eigen::RowVector2d &Q,
                            Eigen::RowVectorXd &G);

    void green_line_integral(const MatX2d &P,
                            const MatX2i &E,
                            const MatX2d &N,
                            const Eigen::VectorXd  &L,
                            const std::vector<int> &ids,
                            const Eigen::RowVector2d &Q,
                            Eigen::RowVectorXd &G,
                            Eigen::RowVectorXd &F);

    void green_line_integral(const Eigen::MatrixX2d &P,
                            const Eigen::MatrixX2i &E,
                            const Eigen::MatrixX2d &N,
                            const Eigen::VectorXd  &L,
                            const std::vector<int> &ids,
                            const Eigen::RowVector2d &Q,
                            Eigen::RowVectorXd &G,
                            Eigen::RowVectorXd &F);


    void green_line_integral(const Eigen::MatrixX2d &P,
                            const Eigen::MatrixX2i &E,
                            const Eigen::MatrixX2d &N,
                            const Eigen::VectorXd  &L,
                            const Eigen::MatrixX2d &Q,
                            Eigen::MatrixXd &G,
                            Eigen::MatrixXd &F);
}

#endif