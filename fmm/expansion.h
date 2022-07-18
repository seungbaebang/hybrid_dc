#ifndef EXPANSION_H
#define EXPANSION_H

#include "Ik.h"
#include "../util/types.h"
#include <vector>
#include <set>

namespace infinite{


    void update_outgoing_from_source_g(const std::vector<bool> &update_pids,
                      const Eigen::VectorXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mg);

    void update_outgoing_from_source_f(const std::vector<bool> &update_pids,
                      const Eigen::VectorXd &mu,
                      const Eigen::VectorXcd &Ni,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mf);


    void update_incoming_from_source_g(const std::vector<bool> &update_pids,
                        const Eigen::VectorXd &sigma,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Ug);

    void update_incoming_from_sourc_f(const std::vector<bool> &update_pids,
                        const Eigen::VectorXd &mu,
                        const Eigen::VectorXcd &Ni,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Uf);


    void incoming_from_incoming(const std::vector<std::vector<int> > &PI,
                        const std::vector<std::vector<int> > &LV, 
                        const Eigen::MatrixX4i &CH,
                        const int &np,
                        const MatXcd_list &Ik_child_list,
                        Eigen::MatrixXcd &Lg,
                        Eigen::MatrixXcd &Lf);

    void incoming_from_incoming(const std::vector<std::vector<int> > &PI,
                        const std::vector<std::vector<int> > &LV, 
                        const Eigen::MatrixX4i &CH,
                        const int &np,
                        const MatXcd_list &Ik_child_list,
                        Eigen::MatrixXcd &Lx);

    void incoming_from_outgoing(const std::vector<std::vector<int> > &PI,
                        const std::vector<std::vector<int> > &Inters,
                        const Eigen::MatrixXcd &Mg,
                        const Eigen::MatrixXcd &Mf,
                        const int &np,
                        const std::vector<VecXcd_list> &Ok_inter_list_list,
                        Eigen::MatrixXcd &Lg,
                        Eigen::MatrixXcd &Lf);

    void incoming_from_outgoing_g(const std::vector<std::vector<int> > &PI,
                        const std::vector<std::vector<int> > &Inters,
                        const Eigen::MatrixXcd &Mg,
                        const int &np,
                        const std::vector<VecXcd_list> &Ok_inter_list_list,
                        Eigen::MatrixXcd &Lg);

    void incoming_from_outgoing_f(const std::vector<std::vector<int> > &PI,
                        const std::vector<std::vector<int> > &Inters,
                        const Eigen::MatrixXcd &Mf,
                        const int &np,
                        const std::vector<VecXcd_list> &Ok_inter_list_list,
                        Eigen::MatrixXcd &Lf);


    void incoming_from_source(const Eigen::VectorXd &sigma,
                        const Eigen::VectorXd &mu,
                        const Eigen::VectorXcd &Ni,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Ug,
                        Eigen::MatrixXcd &Uf);

    void incoming_from_source_g(const Eigen::VectorXd &sigma,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Ug);

    void incoming_from_source_f(const Eigen::VectorXd &mu,
                        const Eigen::VectorXcd &Ni,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Uf);


    void outgoing_from_source(const Eigen::VectorXd &sigma,
                      const Eigen::VectorXd &mu,
                      const Eigen::VectorXcd &Ni,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mg,
                      Eigen::MatrixXcd &Mf);

    void outgoing_from_source_g(const Eigen::VectorXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mg);       

    void outgoing_from_source_f(const Eigen::VectorXd &mu,
                      const Eigen::VectorXcd &Ni,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mf);


    void outgoing_from_outgoing(const std::vector<std::vector<int> > &LV,
                      const std::vector<std::vector<int> > &PI,
                      const MatXcd_list &Ik_child_list,
                      const Eigen::MatrixX4i &CH,
                      const int &np,
                      Eigen::MatrixXcd &Mg,
                      Eigen::MatrixXcd &Mf);

    void outgoing_from_outgoing(const std::vector<std::vector<int> > &LV,
                      const std::vector<std::vector<int> > &PI,
                      const MatXcd_list &Ik_child_list,
                      const Eigen::MatrixX4i &CH,
                      const int &np,
                      Eigen::MatrixXcd &Mx);

}



#endif