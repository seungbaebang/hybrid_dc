
#ifndef COMPUTE_CELL_LIST_H
#define COMPUTE_CELL_LIST_H

#include <Eigen/Core>
#include <vector>
#include <set>
#include <igl/unique.h>

#include "query_cell.h"
#include "../util/util.h"

namespace infinite{
    int tree_neighbour(int index,
                        Eigen::RowVector2i direction,
                        const Eigen::VectorXi &PA,
                        const Eigen::MatrixXi &CH);

    void near_given_level(int neighbour_index, 
                Eigen::RowVector2i direction,
                const int &level,
                const Eigen::VectorXi &LV,
                const Eigen::MatrixXi &CH,
                const std::set<int> &all_leaf_ids,
                std::vector<int> &adjs);

    void get_adj_seperate(int neighbour_index, 
                        Eigen::RowVector2i direction,
                        const Eigen::MatrixXi &CH,
                        const std::set<int> &all_leaf_ids,
                        std::vector<int> &adjs,
                        std::vector<int> &small_seps);

    void compute_cell_list(const Eigen::VectorXi &PA,
                        const Eigen::MatrixXi &CH,
                        std::vector<std::vector<int> > &adjs,
                        std::vector<std::vector<int> > &inters);



    void compute_cell_list(const Eigen::VectorXi &PA,
                       const Eigen::MatrixXi &CH,
                       std::vector<std::vector<int> > &adjs,
                       std::vector<std::vector<int> > &small_seps,
                       std::vector<std::vector<int> > &big_seps,
                       std::vector<std::vector<int> > &inters);

    void compute_cell_list(const Eigen::VectorXi &PA,
                       const Eigen::MatrixXi &CH,
                       std::vector<std::vector<int> > &adjs,
                       std::vector<std::vector<int> > &small_seps,
                       std::vector<std::vector<int> > &big_seps,
                       std::vector<std::vector<int> > &inters,
                       std::vector<std::vector<int> > &uni_adjs);

    void update_cell_list(const Eigen::VectorXi &sub_cells,
                        const Eigen::VectorXi &new_cells,
                        const Eigen::MatrixXi &CH,
                        const Eigen::VectorXi &PA,
                        const Eigen::VectorXi &LV,
                        const std::set<int> &all_leaf_set,
                        std::vector<std::vector<int> > &adjs,
                        std::vector<std::vector<int> > &small_seps,
                        std::vector<std::vector<int> > &big_seps,
                        std::vector<std::vector<int> > &inters,
                        std::vector<std::vector<int> > &uni_adjs);

    void update_cell_list(const Eigen::VectorXi &sub_cells,
                        const Eigen::VectorXi &new_cells,
                        const Eigen::MatrixXi &CH,
                        const Eigen::VectorXi &PA,
                        const Eigen::VectorXi &LV,
                        const std::set<int> &all_leaf_set,
                        std::vector<std::vector<int> > &adjs,
                        std::vector<std::vector<int> > &small_seps,
                        std::vector<std::vector<int> > &big_seps,
                        std::vector<std::vector<int> > &inters,
                        std::vector<std::vector<int> > &uni_adjs,
                        std::vector<std::vector<int> > &inters_a,
                        std::vector<int> &big_sep_updates);
}

#endif