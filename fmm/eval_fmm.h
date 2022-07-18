#ifndef EVAL_FMM_H
#define EVAL_FMM_H

#include <iostream>

#include <Eigen/Core>
#include <igl/slice.h>
#include <igl/PI.h>
#include <igl/get_seconds.h>

// #include "Ik.h"
#include "expansion.h"
#include "compute_cell_list.h"
#include "query_cell.h"
#include "quadtree.h"
#include "../curve/bezier_curves.h"
#include "../curve/bezier_length.h"
#include "../util/edge_indices.h"
#include "../util/mat2d_to_compvec.h"
#include "../bem/green_line_integral.h"
#include "../biem/green_quadrature.h"
#include "../util/types.h"
#include "fmm.h"

namespace infinite{
    
    void eval_fmm_integral(
                  const Eigen::MatrixXd &sigma,
                  const Eigen::MatrixXd &mu,
                  const int &num_expansion,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const std::vector<std::vector<int > > &Q_PEI,
                  const std::vector<std::vector<int > > &Q_QI,
                  const std::vector<std::vector<int> > &levels,
                  const RowVec2d_list_list &Q_LL,
                  const Mat2d_list_list &Q_PP,
                  const Eigen::MatrixXi &Q_CH,
                  const Eigen::VectorXi &Q_LV,
                  const Eigen::MatrixXd &Q_CN, 
                  const std::vector<std::vector<int> > &Q_adjs,
                  const std::vector<std::vector<int> > &Q_small_seps,
                  const std::vector<std::vector<int> > &Q_inters,
                  const std::vector<std::vector<int> > &Q_big_seps,
                  const std::vector<int> &leaf_cells,
                  Eigen::MatrixXd& W);

    void eval_fmm_integral(const Eigen::MatrixX2d &P,
                       const Eigen::MatrixX2i &E,
                       const Eigen::MatrixX2d &C,
                       const Eigen::MatrixX2d &N,
                       const Eigen::VectorXd &L,
                       const Eigen::MatrixX2d &Q,
                       const Eigen::MatrixXd &sigma,
                       const Eigen::MatrixXd &mu,
                       const int &num_expansion,
                       const int &min_pnt_num,
                       const int &max_depth,
                       Eigen::MatrixXd &W);



}

#endif