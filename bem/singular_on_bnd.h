#ifndef SINGULAR_ON_BND_H
#define SINGULAR_ON_BND_H

#include <Eigen/Core>
#include <vector>

#include <igl/slice.h>
#include <igl/PI.h>
#include <igl/mat_min.h>

#include "../util/types.h"
#include "../curve/bezier_length.h"


namespace infinite
{
    // void infinite::singular_on_bnd(const Mat42_list &CPs, 
    //                     const Eigen::VectorXd &XE, 
    //                     const Eigen::VectorXd &XG, 
    //                     Eigen::VectorXd &TOE);


    void singular_on_bnd(const VecXd_list &xe_list, 
                     const Eigen::VectorXd &xc,
                     VecXi_list &sing_id_list,
                     VecXd_list &sing_t_list);

    void singular_on_bnd(const Eigen::VectorXd &xe, 
                     const Eigen::VectorXd &xc,
                     const int &nb,
                     VecXi_list &sing_id_list,
                     VecXd_list &sing_t_list);
                     
    void singular_on_bnd(const Eigen::VectorXd &xe, 
                     const Eigen::VectorXd &xc,
                     const int &nb,
                     std::vector<std::vector<int>> &sing_id_list,
                     std::vector<std::vector<double>> &sing_t_list);

    void singular_on_bnd(const Eigen::VectorXd &xe, 
                        const Eigen::VectorXd &xc,
                        const int &nb,
                        std::vector<std::vector<int>> &sing_edge_list,
                        std::vector<std::vector<int>> &sing_end_list,
                        std::vector<std::vector<double>> &edge_t_list);

    void singular_on_bnd(const Eigen::VectorXd &xe, 
                        const Eigen::VectorXd &xc,
                        const Eigen::VectorXd &S,
                        Eigen::MatrixXd &G,
                        Eigen::MatrixXd &F);
}
#endif