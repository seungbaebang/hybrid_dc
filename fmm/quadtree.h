
#ifndef QUADTREE_H
#define QUADTREE_H

#include <Eigen/Core>
#include <vector>
#include <set>
#include <igl/unique.h>

#include "query_cell.h"
#include "../util/liang_barsky_clipper.h"
#include "../util/types.h"

namespace infinite{

        void determine_update_cells(
            const std::vector<int> &pre_leaf_cells,
            const std::vector<int> &cur_leaf_cells,
            const Eigen::VectorXi &new_cells,
            const Eigen::VectorXi &new_pids,
            const std::vector<std::vector<int > > &pre_Q_PEI,
            const std::vector<std::vector<int > > &cur_Q_PEI,
            std::vector<int> &update_leafs);

        void determine_update_cells(
            const std::vector<int> &pre_leaf_cells,
            const std::vector<int> &cur_leaf_cells,
            const Eigen::VectorXi &new_cells,
            const Eigen::VectorXi &new_pids,
            const std::vector<std::vector<int > > &pre_Q_PEI,
            const std::vector<std::vector<int > > &cur_Q_PEI,
            const std::vector<std::vector<int> > &big_seps,
            const std::vector<int> &big_sep_updates,
            std::vector<int> &update_leafs,
            std::vector<int> &update_big_seps);

        void adap_quadtree(const Eigen::VectorXi &cI,
            const std::vector<std::vector<int> > &cEI,
            const Eigen::VectorXi &qI,
            const Eigen::VectorXi &sub_pids,
            const Eigen::VectorXi &sub_qids,
            const Eigen::VectorXi &new_pids,
            const Eigen::VectorXi &new_qids,
            const Eigen::MatrixXd &nP,
            const Eigen::MatrixXi &nE,
            const Eigen::MatrixXd &nC,
            const Eigen::MatrixXd &nQ,
            const int& max_depth,
            const int& min_pnt_num,
            std::vector<std::vector<int> > &P_I,
            std::vector<std::vector<int> > &P_EI,
            std::vector<std::vector<int> > &Q_I,
            RowVec2d_list_list &P_LL,
            Mat2d_list_list &P_PP,
            Eigen::MatrixXi &CH,
            Eigen::VectorXi &PA,
            Eigen::VectorXi &LV,
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W,
            Eigen::VectorXi &SC);

    void adap_quadtree(const Eigen::VectorXi &cI,
            const std::vector<std::vector<int> > &cEI,
            const Eigen::VectorXi &sub_ids,
            const Eigen::MatrixXd &nP,
            const Eigen::MatrixXi &nE,
            const Eigen::MatrixXd &nC,
            const Eigen::MatrixXd &nQ,
            const Eigen::VectorXi &new_ids,
            const int& max_depth,
            const int& min_pnt_num,
            std::vector<std::vector<int> > &P_I,
            std::vector<std::vector<int> > &P_EI,
            std::vector<std::vector<int> > &Q_I,
            RowVec2d_list_list &P_LL,
            Mat2d_list_list &PP,
            Eigen::MatrixXi &CH,
            Eigen::VectorXi &PA,
            Eigen::VectorXi &LV,
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W,
            Eigen::VectorXi &SC);


    void adap_quadtree(const Eigen::VectorXi &cI,
            const Eigen::VectorXi &sub_ids,
            const Eigen::MatrixXd &nC,
            const Eigen::VectorXi &new_ids,
            const int& max_depth,
            const int& min_pnt_num,
            std::vector<std::vector<int> > &P_I,
            Eigen::MatrixXi &CH,
            Eigen::VectorXi &PA,
            Eigen::VectorXi &LV,
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W,
            Eigen::VectorXi &SC);

    void quadtree(const Eigen::MatrixXd &P, 
             const Eigen::MatrixXi &E,
             const Eigen::MatrixXd &C, 
             const Eigen::MatrixXd &Q,
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& max_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &P_EI,
             std::vector<std::vector<int> > &Q_I,
             RowVec2d_list_list &P_LL,
             Mat2d_list_list &cPE,
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::VectorXi &LV,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W);


    void quadtree(const Eigen::MatrixXd &P, 
            const Eigen::MatrixXi &E,
            const Eigen::MatrixXd &C, 
            const Eigen::RowVector2d& minP, 
            const Eigen::RowVector2d& maxP,
            const int& max_depth,
            const int& min_pnt_num,
            std::vector<std::vector<int> > &P_I, 
            std::vector<std::vector<int> > &P_EI,
            RowVec2d_list_list &P_LL,
            Mat2d_list_list &cPE,
            Eigen::MatrixXi &CH,
            Eigen::VectorXi &PA,
            Eigen::VectorXi &LV,
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W);



    void quadtree(const Eigen::MatrixXd &P, 
                const Eigen::MatrixXd &Q,
                const Eigen::RowVector2d& minP, 
                const Eigen::RowVector2d& maxP,
                const int& max_depth,
                const int& min_pnt_num,
                std::vector<std::vector<int> > &P_I, 
                std::vector<std::vector<int> > &Q_I, 
                Eigen::MatrixXi &CH,
                Eigen::VectorXi &PA,
                std::vector<std::vector<int> > &Lv,
                Eigen::MatrixXd &CN, 
                Eigen::VectorXd &W);

    void quadtree(const Eigen::MatrixXd &P, 
                const Eigen::RowVector2d& minP, 
                const Eigen::RowVector2d& maxP,
                const int& max_depth,
                const int& min_pnt_num,
                std::vector<std::vector<int> > &point_indices, 
                Eigen::MatrixXi &CH,
                Eigen::VectorXi &PA,
                std::vector<std::vector<int> > &Lv,
                Eigen::MatrixXd &CN, 
                Eigen::VectorXd &W);

 
  
}
#endif