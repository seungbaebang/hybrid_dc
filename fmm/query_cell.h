#ifndef QUERY_CELL_H
#define QUERY_CELL_H

#include <Eigen/Core>
#include "../util/types.h"
#include "../util/liang_barsky_clipper.h"

namespace infinite{

    Eigen::RowVector2i get_node_descent(const int &index);

    Eigen::RowVector2i query_direction(int index, 
                            int neighbour_index, 
                            const Eigen::VectorXi &PA, 
                            const Eigen::VectorXi &LV);

    void update_cells(const Eigen::MatrixX4i &CH, 
                    const Eigen::VectorXi &PA,
                    const Eigen::MatrixXd &CN,
                    const Eigen::VectorXd &W,
                    const Eigen::Matrix2d &pp,
                    const int &pid,
                    std::vector<std::vector<int> > &P_EI,
                    RowVec2d_list_list &P_LL,
                    Mat2d_list_list &P_PP,
                    std::vector<int> qEI_list);


    void update_cells(const Eigen::MatrixX4i &CH, 
                    const Eigen::VectorXi &PA,
                    const Eigen::MatrixXd &CN,
                    const Eigen::VectorXd &W,
                    const Eigen::Matrix2d &pp,
                    const int &pid,
                    std::vector<std::vector<int> > &P_EI,
                    RowVec2d_list_list &P_LL,
                    Mat2d_list_list &P_PP);

    int query_cell(const Eigen::MatrixX4i &CH, 
               const Eigen::MatrixXd &CN,
               const Eigen::VectorXd &W,
               const Eigen::RowVector2d &q);

    int query_cell(const Eigen::MatrixX4i &CH, 
                const Eigen::VectorXcd &CNi,
                const std::complex<double> &q);
}
#endif