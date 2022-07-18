#ifndef FMM_H
#define FMM_H

#include <iostream>
#include <math.h> 

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <Eigen/Sparse>

#include <igl/unique.h>
#include <igl/slice.h>
#include <igl/PI.h>
#include <igl/get_seconds.h>
#include <igl/slice_into.h>

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
#include "../bem/singular_on_bnd.h"
#include "../biem/interpolate_density.h"
#include "../util/util.h"

typedef Eigen::Matrix<double,4,2> Mat42;

namespace infinite{
    
void precompute_expansions_cell_dependent
                    (const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PEI,
                     const RowVec2d_list_list &Q_LL,
                     const Mat2d_list_list &Q_PP,
                     const Eigen::MatrixXi &Q_CH,
                     const Eigen::VectorXi &Q_LV,
                     const Eigen::MatrixXd &Q_CN, 
                     const std::vector<std::vector<int> > &Q_big_seps,
                     const std::vector<std::vector<int> > &Q_inters,
                     std::vector<VecXcd_list> &Ik_out_list_list,
                     std::vector<VecXcd_list> &Ok_inter_list_list,
                     MatXcd_list &Ik_child_list,
                     std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                     VecXcd_list &lw_list);


void precompute_expansions_query_dependent
                    (const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const Eigen::MatrixX2d &Q,
                     const Eigen::VectorXi &qI,
                     const VecXi_list &sing_id_list,
                     const VecXd_list &sing_t_list,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PEI,
                     const std::vector<std::vector<int > > &Q_QI,
                     const RowVec2d_list_list &Q_LL,
                     const Mat2d_list_list &Q_PP,
                     const Eigen::MatrixXi &Q_CH,
                     const Eigen::MatrixXd &Q_CN, 
                     const std::vector<std::vector<int> > &Q_adjs,
                     const std::vector<std::vector<int> > &Q_small_seps,
                     const std::vector<int> &leaf_cells,
                     std::vector<RowVecXd_list> &G_list,
                     std::vector<RowVecXd_list> &F_list,
                     std::vector<VecXi_list> &nI_list,
                     std::vector<VecXcd_list> &Ok_l3_list,
                     VecXcd_list &Ik_inc_list);



void collect_expansions_g_vec(const int &num_expansion,
                      const Eigen::VectorXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps,
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      Eigen::MatrixXcd &Mg,
                      Eigen::MatrixXcd &Ug,
                      Eigen::MatrixXcd &Lg,
                      Eigen::MatrixXcd &MULg);

void collect_expansions_g(const int &num_expansion,
                      const Eigen::MatrixXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps, 
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      MatXcd_list &Mg_list,
                      MatXcd_list &Ug_list,
                      MatXcd_list &Lg_list,
                      MatXcd_list &MULg_list);

void collect_expansions_f(const Eigen::VectorXcd &Ni,
                      const int &num_expansion,
                      const Eigen::MatrixXd &mu,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps, 
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      MatXcd_list &Mf_list,
                      MatXcd_list &Uf_list,
                      MatXcd_list &Lf_list,
                      MatXcd_list &MULf_list);

   


void eval_f(const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<RowVecXd_list> &F_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const MatXcd_list &Mf_list,
                    const MatXcd_list &MULf_list,
                    const Eigen::MatrixXd &mu,
                    Eigen::MatrixXd &Wf_far,
                    Eigen::MatrixXd &Wf_near);


void eval_g(const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<RowVecXd_list> &G_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const MatXcd_list &Mg_list,
                    const MatXcd_list &MULg_list,
                    const Eigen::MatrixXd &sigma,
                    Eigen::MatrixXd &Wg_far,
                    Eigen::MatrixXd &Wg_near);

void eval_g_vec(const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<RowVecXd_list> &G_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const Eigen::MatrixXcd &Mg,
                    const Eigen::MatrixXcd &MULg,
                    const Eigen::VectorXd &sigma,
                    Eigen::VectorXd &Wg_far,
                    Eigen::VectorXd &Wg_near);


    void pre_fmm(const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const Eigen::MatrixX2d &Q,
                     const Eigen::VectorXi &qI,
                     const VecXi_list &sing_id_list,
                     const VecXd_list &sing_t_list,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PEI,
                     const std::vector<std::vector<int > > &Q_QI,
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
                     std::vector<VecXcd_list> &Ik_out_list_list,
                     std::vector<VecXcd_list> &Ok_inter_list_list,
                     MatXcd_list &Ik_child_list,
                     std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                     VecXcd_list &lw_list,
                     std::vector<RowVecXd_list> &G_list,
                     std::vector<RowVecXd_list> &F_list,
                     std::vector<VecXi_list> &nI_list,
                     std::vector<VecXcd_list> &Ok_l3_list,
                     VecXcd_list &Ik_inc_list);



    void forward_fmm_f(const Eigen::VectorXcd &Ni,
                      const int &num_expansion,
                      const Eigen::VectorXi &qI,
                      const Eigen::MatrixXd &mu,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_adjs,
                      const std::vector<std::vector<int> > &Q_small_seps,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps, 
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      const std::vector<RowVecXd_list> &F_list,
                      const std::vector<VecXi_list> &nI_list,
                      const std::vector<VecXcd_list> &Ok_l3_list,
                      const VecXcd_list &Ik_inc_list,
                      Eigen::MatrixXd &Wf);

    void forward_fmm_g(const int &num_expansion,
                      const Eigen::VectorXi &qI,
                      const Eigen::MatrixXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_adjs,
                      const std::vector<std::vector<int> > &Q_small_seps,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps, 
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      const std::vector<RowVecXd_list> &G_list,
                      const std::vector<VecXi_list> &nI_list,
                      const std::vector<VecXcd_list> &Ok_l3_list,
                      const VecXcd_list &Ik_inc_list,
                      Eigen::MatrixXd &Wg);

    void forward_fmm_g_vec(const int &num_expansion,
                      const Eigen::VectorXi &qI,
                      const Eigen::VectorXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_adjs,
                      const std::vector<std::vector<int> > &Q_small_seps,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps, 
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      const std::vector<RowVecXd_list> &G_list,
                      const std::vector<VecXi_list> &nI_list,
                      const std::vector<VecXcd_list> &Ok_l3_list,
                      const VecXcd_list &Ik_inc_list,
                      Eigen::VectorXd &Wg);

/////////////////////////////////////////////////////////////////////////

void update_expansions_cell_dependent
                    (const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PEI,
                     const RowVec2d_list_list &Q_LL,
                     const Mat2d_list_list &Q_PP,
                     const Eigen::MatrixXi &Q_CH,
                     const Eigen::VectorXi &Q_LV,
                     const Eigen::MatrixXd &Q_CN, 
                     const Eigen::VectorXi &sub_cells,
                     const Eigen::VectorXi &new_cells,
                     const std::vector<std::vector<int> > &Q_big_seps,
                     const std::vector<std::vector<int> > &Q_inters_a,
                     const std::vector<int> &update_leafs,
                     const std::vector<int> &update_l4s,
                     std::vector<VecXcd_list> &Ik_out_list_list,
                     std::vector<VecXcd_list> &Ok_inter_list_list,
                     MatXcd_list &Ik_child_list,
                     std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                     VecXcd_list &lw_list);


void update_expansions_query_dependent
                    (const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const Eigen::MatrixX2d &Q,
                     const VecXi_list &sing_id_list,
                     const VecXd_list &sing_t_list,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PEI,
                     const std::vector<std::vector<int > > &Q_QI,
                     const RowVec2d_list_list &Q_LL,
                     const Mat2d_list_list &Q_PP,
                     const Eigen::MatrixXi &Q_CH,
                     const Eigen::MatrixXd &Q_CN, 
                     const Eigen::VectorXi &new_cells,
                     const std::vector<std::vector<int> > &Q_adjs,
                     const std::vector<std::vector<int> > &Q_small_seps,
                     const std::vector<int> &leaf_cells,
                     const Eigen::VectorXi &new_pids,
                     const Eigen::VectorXi &sub_qids,
                     const Eigen::VectorXi &new_qids,
                     std::vector<RowVecXd_list> &G_list,
                     std::vector<RowVecXd_list> &F_list,
                     std::vector<VecXi_list> &nI_list,
                     std::vector<VecXcd_list> &Ok_l3_list,
                     VecXcd_list &Ik_inc_list);


void update_collecting_expansions_g_vec
                      (const int &num_expansion,
                      const Eigen::VectorXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps,
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      const std::vector<bool> &update_pids,
                      Eigen::MatrixXcd &Mg,
                      Eigen::MatrixXcd &Ug,
                      Eigen::MatrixXcd &Lg,
                      Eigen::MatrixXcd &MULg);



void update_collecting_expansions_g
                      (const int &num_expansion,
                      const Eigen::MatrixXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps,
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      const std::vector<bool> &update_pids,
                      MatXcd_list &Mg_list,
                      MatXcd_list &Ug_list,
                      MatXcd_list &Lg_list,
                      MatXcd_list &MULg_list);


void update_collecting_expansions_f
                      (const Eigen::VectorXcd &Ni,
                      const int &num_expansion,
                      const Eigen::MatrixXd &mu,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int > > &levels,
                      const std::vector<std::vector<int > > &Q_PEI,
                      const std::vector<std::vector<int> > &Q_inters,
                      const std::vector<std::vector<int> > &Q_big_seps,
                      const Eigen::MatrixXi &Q_CH,
                      const std::vector<VecXcd_list> &Ik_out_list_list,
                      const std::vector<VecXcd_list> &Ok_inter_list_list,
                      const MatXcd_list &Ik_child_list,
                      const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                      const std::vector<bool> &update_pids,
                      MatXcd_list &Mf_list,
                      MatXcd_list &Uf_list,
                      MatXcd_list &Lf_list,
                      MatXcd_list &MULf_list);



void update_eval_g_vec(const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const Eigen::VectorXi &sol_qids,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<RowVecXd_list> &G_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const Eigen::MatrixXcd &Mg,
                    const Eigen::MatrixXcd &MULg,
                    const Eigen::VectorXd &sigma,
                    const std::vector<bool> &update_pids,
                    Eigen::VectorXd &Wg_far,
                    Eigen::VectorXd &Wg_near);


void update_eval_g(const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const Eigen::VectorXi &sol_qids,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<RowVecXd_list> &G_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const MatXcd_list &Mg_list,
                    const MatXcd_list &MULg_list,
                    const Eigen::MatrixXd &sigma,
                    const std::vector<bool> &update_pids,
                    Eigen::MatrixXd &Wg_far,
                    Eigen::MatrixXd &Wg_near);


void update_eval_f(const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const Eigen::VectorXi &sol_qids,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<RowVecXd_list> &F_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const MatXcd_list &Mf_list,
                    const MatXcd_list &MULf_list,
                    const Eigen::MatrixXd &mu,
                    const std::vector<bool> &update_pids,
                    Eigen::MatrixXd &Wf_far,
                    Eigen::MatrixXd &Wf_near);

void update_pre_fmm_hybrid(const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const Eigen::MatrixX2d &Q,
                     const VecXi_list &sing_id_list,
                     const VecXd_list &sing_t_list,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PEI,
                     const std::vector<std::vector<int > > &Q_QI,
                     const RowVec2d_list_list &Q_LL,
                     const Mat2d_list_list &Q_PP,
                     const Eigen::MatrixXi &Q_CH,
                     const Eigen::VectorXi &Q_LV,
                     const Eigen::MatrixXd &Q_CN, 
                     const Eigen::VectorXi &sub_cells,
                     const Eigen::VectorXi &new_cells,
                     const std::vector<int> &leaf_cells,
                     const std::vector<std::vector<int> > &Q_adjs,
                     const std::vector<std::vector<int> > &Q_small_seps,
                     const std::vector<std::vector<int> > &Q_big_seps,
                     const std::vector<std::vector<int> > &Q_inters_a,
                     const std::vector<int> &update_leafs,
                     const std::vector<int> &update_l4s,
                     const Eigen::VectorXi &new_pids,
                     const Eigen::VectorXi &sub_qids,
                     const Eigen::VectorXi &new_qids,
                     std::vector<VecXcd_list> &Ik_out_list_list,
                     std::vector<VecXcd_list> &Ok_inter_list_list,
                     MatXcd_list &Ik_child_list,
                     std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                     VecXcd_list &lw_list,
                     std::vector<RowVecXd_list> &G_list,
                     std::vector<RowVecXd_list> &F_list,
                     std::vector<VecXi_list> &nI_list,
                     std::vector<VecXcd_list> &Ok_l3_list,
                     VecXcd_list &Ik_inc_list);

void update_forward_fmm_f(const Eigen::VectorXcd &Ni,
                    const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const Eigen::VectorXi &sol_qids,
                    const Eigen::MatrixXd &mu,
                    const std::vector<int> &leaf_cells,
                    const std::vector<std::vector<int > > &levels,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<std::vector<int> > &Q_inters,
                    const std::vector<std::vector<int> > &Q_big_seps, 
                    const Eigen::MatrixXi &Q_CH,
                    const std::vector<VecXcd_list> &Ik_out_list_list,
                    const std::vector<VecXcd_list> &Ok_inter_list_list,
                    const MatXcd_list &Ik_child_list,
                    const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                    const std::vector<RowVecXd_list> &F_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const std::vector<bool> &update_pids,
                    Eigen::MatrixXd &Wf);

void update_forward_fmm_g(const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const Eigen::VectorXi &sol_qids,
                    const Eigen::MatrixXd &sigma,
                    const std::vector<int> &leaf_cells,
                    const std::vector<std::vector<int > > &levels,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<std::vector<int> > &Q_inters,
                    const std::vector<std::vector<int> > &Q_big_seps, 
                    const Eigen::MatrixXi &Q_CH,
                    const std::vector<VecXcd_list> &Ik_out_list_list,
                    const std::vector<VecXcd_list> &Ok_inter_list_list,
                    const MatXcd_list &Ik_child_list,
                    const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                    const std::vector<RowVecXd_list> &G_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const std::vector<bool> &update_pids,
                    Eigen::MatrixXd &Wg);

void update_forward_fmm_g_vec(
                    const int &num_expansion,
                    const Eigen::VectorXi &qI,
                    const Eigen::VectorXi &sol_qids,
                    const Eigen::VectorXd &sigma,
                    const std::vector<int> &leaf_cells,
                    const std::vector<std::vector<int > > &levels,
                    const std::vector<std::vector<int > > &Q_PEI,
                    const std::vector<std::vector<int> > &Q_adjs,
                    const std::vector<std::vector<int> > &Q_small_seps,
                    const std::vector<std::vector<int> > &Q_inters,
                    const std::vector<std::vector<int> > &Q_big_seps, 
                    const Eigen::MatrixXi &Q_CH,
                    const std::vector<VecXcd_list> &Ik_out_list_list,
                    const std::vector<VecXcd_list> &Ok_inter_list_list,
                    const MatXcd_list &Ik_child_list,
                    const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                    const std::vector<RowVecXd_list> &G_list,
                    const std::vector<VecXi_list> &nI_list,
                    const std::vector<VecXcd_list> &Ok_l3_list,
                    const VecXcd_list &Ik_inc_list,
                    const std::vector<bool> &update_pids,
                    Eigen::VectorXd &Wg);
}
#endif