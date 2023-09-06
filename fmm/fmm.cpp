#include "fmm.h"


void infinite::precompute_expansions_cell_dependent_noclip
                    (const Eigen::MatrixX2d &P,
                     const Eigen::MatrixX2i &E,
                     const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PI,
                     const Eigen::MatrixXi &Q_CH,
                     const Eigen::VectorXi &Q_LV,
                     const Eigen::MatrixXd &Q_CN, 
                     const std::vector<std::vector<int> > &Q_inters,
                     const std::vector<std::vector<int> > &Q_big_seps,
                     std::vector<VecXcd_list> &Ik_out_list_list,
                     std::vector<VecXcd_list> &Ok_inter_list_list,
                     MatXcd_list &Ik_child_list,
                     std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list)
{
  double PI_4 = 4*igl::PI;
  double PI_2 = 2*igl::PI;


  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  Ok_inter_list_list.resize(Q_inters.size());
  #pragma omp parallel for
  for(int i=0; i<Q_inters.size(); i++){
    VecXcd_list Ok_list(Q_inters[i].size());
    for(int j=0; j<Q_inters[i].size(); j++){
      Eigen::VectorXcd Ok = Eigen::VectorXcd::Zero(2*num_expansion);
      compute_Ok(Q_CNi[i]-Q_CNi[Q_inters[i][j]],2*num_expansion,Ok);
      Ok_list[j]=Ok;
    }
    Ok_inter_list_list[i]=Ok_list;
  }

  Ik_child_list.resize(Q_CH.rows());
  #pragma omp parallel for
  for(int i=0; i<Q_CH.rows();i++){
    // if(Q_PI[i].size()==0)
    //   continue;
    if(Q_CH(i,0)<0)
      continue;
    Eigen::MatrixXcd Ik(4,num_expansion);
    for(int ci=0; ci<4; ci++){
      compute_Ik(Q_CNi[Q_CH(i,ci)]-Q_CNi[i],num_expansion,Ik,ci);
    }
    Ik_child_list[i]=Ik;
  }
  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }

  VecXcd_list lw_list(Q_PI.size());

  Ik_out_list_list.resize(Q_PI.size());
  // std::vector<Eigen::VectorXcd> lw_list(leaf_cells.size());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int li = leaf_cells[i];    
    std::complex<double> cni = Q_CNi(li);
    VecXcd_list Ik_list(Q_PI[li].size());
    Eigen::VectorXcd lw(Q_PI[li].size());
    for(int ii=0; ii<Q_PI[li].size(); ii++){
      //assumes tiny Q_L is already removed
      int pid = Q_PI[li][ii];
      std::complex<double> za(P(E(pid,0),0),P(E(pid,0),1));
      std::complex<double> zb(P(E(pid,1),0),P(E(pid,1),1));
      const std::complex<double> &w = (zb-za)/std::norm(zb-za);

      lw(ii)=L(pid)*std::conj(w);

      Eigen::VectorXcd Ik1 = Eigen::VectorXcd::Zero(num_expansion+1);
      Eigen::VectorXcd Ik2 = Eigen::VectorXcd::Zero(num_expansion+1);
      compute_Ik(zb-cni,num_expansion+1,Ik1);
      compute_Ik(za-cni,num_expansion+1,Ik2);

      Ik_list[ii] = lw(ii)*(Ik1.tail(num_expansion)-Ik2.tail(num_expansion));
    }
    Ik_out_list_list[li]=Ik_list;
    lw_list[li]=lw;
  }

  Ok_inc_list_list.resize(Q_big_seps.size());
  #pragma omp parallel for
  for(int i=0; i<Q_big_seps.size(); i++){
    const std::complex<double> &cni = Q_CNi(i);

    Ok_inc_list_list[i].resize(Q_big_seps[i].size());
    for(int ii=0; ii<Q_big_seps[i].size(); ii++){
      int l4_cell = Q_big_seps[i][ii];
      Ok_inc_list_list[i][ii].resize(Q_PI[l4_cell].size());
      for(int jj=0; jj<Q_PI[l4_cell].size(); jj++){

        int pid = Q_PI[l4_cell][jj];

        std::complex<double> za(P(E(pid,0),0),P(E(pid,0),1));
        std::complex<double> zb(P(E(pid,1),0),P(E(pid,1),1));

        Eigen::VectorXcd Ok1 = Eigen::VectorXcd::Zero(num_expansion-1);
        Eigen::VectorXcd Ok2 = Eigen::VectorXcd::Zero(num_expansion-1);
        compute_Ok_no_log(zb-cni,num_expansion-1,Ok1);
        compute_Ok_no_log(za-cni,num_expansion-1,Ok2);


        Eigen::VectorXcd Ok = Eigen::VectorXcd::Zero(num_expansion+1);
        Ok(0) = -green_line_integral(P,E,pid,L(pid),Q_CN.row(i));
        // Ok(0) = -green_line_integral(Pe,L(pid),Q_CN.row(i));
        Ok(1) = -lw_list[l4_cell](jj)*(log((zb-cni)/(za-cni)));
        Ok.tail(num_expansion-1) = lw_list[l4_cell](jj)*(Ok1-Ok2);
        Ok_inc_list_list[i][ii][jj]=Ok;
      }
    }
  }

}


void infinite::precompute_expansions_cell_dependent
                    (const Eigen::MatrixX2d &P,
                     const Eigen::MatrixX2i &E,
                     const Eigen::MatrixX2d &N,
                     const Eigen::VectorXd &L,
                     const int &num_expansion,
                     const std::vector<std::vector<int > > &Q_PI,
                     const Eigen::MatrixXi &Q_CH,
                     const Eigen::VectorXi &Q_LV,
                     const Eigen::MatrixXd &Q_CN, 
                     const std::vector<std::vector<int> > &Q_inters,
                     std::vector<VecXcd_list> &Ik_out_list_list,
                     std::vector<VecXcd_list> &Ok_inter_list_list,
                     MatXcd_list &Ik_child_list)
{
  double PI_4 = 4*igl::PI;
  double PI_2 = 2*igl::PI;


  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  Ok_inter_list_list.resize(Q_inters.size());
  #pragma omp parallel for
  for(int i=0; i<Q_inters.size(); i++){
    VecXcd_list Ok_list(Q_inters[i].size());
    for(int j=0; j<Q_inters[i].size(); j++){
      Eigen::VectorXcd Ok = Eigen::VectorXcd::Zero(2*num_expansion);
      compute_Ok(Q_CNi[i]-Q_CNi[Q_inters[i][j]],2*num_expansion,Ok);
      Ok_list[j]=Ok;
    }
    Ok_inter_list_list[i]=Ok_list;
  }

  Ik_child_list.resize(Q_CH.rows());
  #pragma omp parallel for
  for(int i=0; i<Q_CH.rows();i++){
    // if(Q_PI[i].size()==0)
    //   continue;
    if(Q_CH(i,0)<0)
      continue;
    Eigen::MatrixXcd Ik(4,num_expansion);
    for(int ci=0; ci<4; ci++){
      compute_Ik(Q_CNi[Q_CH(i,ci)]-Q_CNi[i],num_expansion,Ik,ci);
    }
    Ik_child_list[i]=Ik;
  }
  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }

  Ik_out_list_list.resize(Q_PI.size());
  // std::vector<Eigen::VectorXcd> lw_list(leaf_cells.size());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int li = leaf_cells[i];    
    std::complex<double> cni = Q_CNi(li);
    VecXcd_list Ik_list(Q_PI[li].size());
    for(int ii=0; ii<Q_PI[li].size(); ii++){

      //assumes tiny Q_L is already removed
      int pid = Q_PI[li][ii];
      std::complex<double> za(P(E(pid,0),0),P(E(pid,0),1));
      std::complex<double> zb(P(E(pid,1),0),P(E(pid,1),1));
      const std::complex<double> &w = (zb-za)/std::norm(zb-za);
      Eigen::VectorXcd Ik1 = Eigen::VectorXcd::Zero(num_expansion+1);
      Eigen::VectorXcd Ik2 = Eigen::VectorXcd::Zero(num_expansion+1);
      compute_Ik(zb-cni,num_expansion+1,Ik1);
      compute_Ik(za-cni,num_expansion+1,Ik2);

      Ik_list[ii] = L(pid)*std::conj(w)*(Ik1.tail(num_expansion)-Ik2.tail(num_expansion));
    }
    Ik_out_list_list[li]=Ik_list;
  }

}




void infinite::precompute_expansions_cell_dependent
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
                     VecXcd_list &lw_list)
{
  double PI_4 = 4*igl::PI;
  double PI_2 = 2*igl::PI;
  std::vector<std::vector<double> > Q_L(Q_LL.size());
  #pragma omp parallel for
  for(int i=0; i<Q_LL.size(); i++){
    Q_L[i].resize(Q_LL[i].size());
    for(int j=0; j<Q_LL[i].size(); j++){
      Q_L[i][j]=Q_LL[i][j](1)-Q_LL[i][j](0);
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  Ok_inter_list_list.resize(Q_inters.size());
  #pragma omp parallel for
  for(int i=0; i<Q_inters.size(); i++){
    VecXcd_list Ok_list(Q_inters[i].size());
    for(int j=0; j<Q_inters[i].size(); j++){
      Eigen::VectorXcd Ok = Eigen::VectorXcd::Zero(2*num_expansion);
      compute_Ok(Q_CNi[i]-Q_CNi[Q_inters[i][j]],2*num_expansion,Ok);
      Ok_list[j]=Ok;
    }
    Ok_inter_list_list[i]=Ok_list;
  }

  Ik_child_list.resize(Q_CH.rows());
  #pragma omp parallel for
  for(int i=0; i<Q_CH.rows();i++){
    if(Q_PEI[i].size()==0)
      continue;
    if(Q_CH(i,0)<0)
      continue;
    Eigen::MatrixXcd Ik(4,num_expansion);
    for(int ci=0; ci<4; ci++){
      compute_Ik(Q_CNi[Q_CH(i,ci)]-Q_CNi[i],num_expansion,Ik,ci);
    }
    Ik_child_list[i]=Ik;
  }
  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }

  lw_list.resize(Q_PEI.size());
  // std::vector<Eigen::VectorXcd> lw_list(leaf_cells.size());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int li = leaf_cells[i];    
    Eigen::VectorXcd lw(Q_PEI[li].size());
    for(int ii=0; ii<Q_PEI[li].size(); ii++){
      if(Q_L[li][ii]<1e-15){
        lw(ii)=0;
        continue;
      }
      //assumes tiny Q_L is already removed
      int pid = Q_PEI[li][ii];
      Eigen::Matrix2d Pe = Q_PP[li][ii];
      std::complex<double> za(Pe(0,0),Pe(0,1));
      std::complex<double> zb(Pe(1,0),Pe(1,1));
      const std::complex<double> &w = (zb-za)/std::norm(zb-za);
      lw(ii)=Q_L[li][ii]*L(pid)*std::conj(w);
    }
    lw_list[li]=lw;
  }

  Ik_out_list_list.resize(Q_PEI.size());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int li = leaf_cells[i];
    std::complex<double> cni = Q_CNi(li);
    VecXcd_list Ik_list(Q_PEI[li].size());
    for(int ii=0; ii<Q_PEI[li].size(); ii++){
      if(Q_L[li][ii]<1e-15){
        Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
        Ik_list[ii]=Ik;
        continue;
      }
      Eigen::Matrix2d Pe = Q_PP[li][ii];
      std::complex<double> za(Pe(0,0),Pe(0,1));
      std::complex<double> zb(Pe(1,0),Pe(1,1));
      Eigen::VectorXcd Ik1 = Eigen::VectorXcd::Zero(num_expansion+1);
      Eigen::VectorXcd Ik2 = Eigen::VectorXcd::Zero(num_expansion+1);
      compute_Ik(zb-cni,num_expansion+1,Ik1);
      compute_Ik(za-cni,num_expansion+1,Ik2);
      Ik_list[ii] = lw_list[li](ii)*(Ik1.tail(num_expansion)-Ik2.tail(num_expansion));
    }
    Ik_out_list_list[li]=Ik_list;
  }

  Ok_inc_list_list.resize(Q_big_seps.size());
  #pragma omp parallel for
  for(int i=0; i<Q_big_seps.size(); i++){
    const std::complex<double> &cni = Q_CNi(i);

    Ok_inc_list_list[i].resize(Q_big_seps[i].size());
    for(int ii=0; ii<Q_big_seps[i].size(); ii++){
      int l4_cell = Q_big_seps[i][ii];
      Ok_inc_list_list[i][ii].resize(Q_PEI[l4_cell].size());
      for(int jj=0; jj<Q_PEI[l4_cell].size(); jj++){

        int pid = Q_PEI[l4_cell][jj];

        Eigen::Matrix2d Pe = Q_PP[l4_cell][jj];
        std::complex<double> za(Pe(0,0),Pe(0,1));
        std::complex<double> zb(Pe(1,0),Pe(1,1));

        double l = Q_L[l4_cell][jj]*L(pid);

        Eigen::VectorXcd Ok1 = Eigen::VectorXcd::Zero(num_expansion-1);
        Eigen::VectorXcd Ok2 = Eigen::VectorXcd::Zero(num_expansion-1);
        compute_Ok_no_log(zb-cni,num_expansion-1,Ok1);
        compute_Ok_no_log(za-cni,num_expansion-1,Ok2);


        Eigen::VectorXcd Ok = Eigen::VectorXcd::Zero(num_expansion+1);
        Ok(0) = -green_line_integral(Pe,l,Q_CN.row(i));
        Ok(1) = -lw_list[l4_cell](jj)*(log((zb-cni)/(za-cni)));
        Ok.tail(num_expansion-1) = lw_list[l4_cell](jj)*(Ok1-Ok2);
        Ok_inc_list_list[i][ii][jj]=Ok;
      }
    }
  }
}


void infinite::precompute_expansions_query_dependent
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
                     VecXcd_list &Ik_inc_list)
{
  double PI_4 = 4*igl::PI;
  double PI_2 = 2*igl::PI;

  Eigen::VectorXcd Qi = mat2d_to_compvec(Q);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  std::vector<std::vector<double> > Q_L(Q_LL.size());
  for(int i=0; i<Q_LL.size(); i++){
    Q_L[i].resize(Q_LL[i].size());
    for(int j=0; j<Q_LL[i].size(); j++){
      Q_L[i][j]=Q_LL[i][j](1)-Q_LL[i][j](0);
    }
  }
  G_list.resize(Qi.size());
  F_list.resize(Qi.size());

  nI_list.resize(Qi.size());
  Ok_l3_list.resize(Qi.size());
  Ik_inc_list.resize(Qi.size());
  
  for(int i=0; i<Qi.size(); i++){
    std::complex<double> qi = Qi(i);
    int cell_id = qI(i);
    std::vector<int> adj_ids = Q_adjs[cell_id];
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    adj_ids.emplace_back(cell_id);

    G_list[i].resize(adj_ids.size());
    F_list[i].resize(adj_ids.size());
    nI_list[i].resize(adj_ids.size());
    Eigen::VectorXi sI=sing_id_list[i];
    Eigen::VectorXd sT=sing_t_list[i];

    for(int j=0; j<adj_ids.size(); j++){
      std::vector<int> near_ids = Q_PEI[adj_ids[j]];
      if(near_ids.size()==0)
        continue;
      Mat2d_list PEs = Q_PP[adj_ids[j]];
      std::vector<double> Ls = Q_L[adj_ids[j]];
      RowVec2d_list LL = Q_LL[adj_ids[j]];
      Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());
      nI_list[i][j]=nI;

      Eigen::MatrixX2d N_n=igl::slice(N,nI,1);
      Eigen::VectorXd L_n =igl::slice(L,nI);

      //ratio of length
      Eigen::VectorXd Lsv=Eigen::Map<Eigen::VectorXd>(Ls.data(),Ls.size());
      Lsv = Lsv.array()*L_n.array();

      Eigen::RowVectorXd G;
      Eigen::RowVectorXd F;
      green_line_integral(PEs,N_n,Lsv,Q.row(i),G,F);

      for(int ii=0; ii<sI.size(); ii++){
        for(int jj=0; jj<nI.size(); jj++){
          if(sI(ii)==nI(jj)){
            if((LL[jj](0)-1e-1)<sT(ii) && (LL[jj](1)+1e-1)>sT(ii)){
              double s_edge = Lsv(jj);
              double toe = (sT(ii)-LL[jj](0))/(LL[jj](1)-LL[jj](0))*s_edge;
              G(jj) = ((toe-s_edge)*log((toe-s_edge)*(toe-s_edge))
                          -toe*log(toe*toe)+2*s_edge)/PI_4;
              F(jj) = 0;
            }
          }
        }
      }
      G_list[i][j]=G;
      F_list[i][j]=F;
    }

    std::complex<double> cni = Q_CNi(cell_id);

    Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
    compute_Ik(qi-cni, num_expansion, Ik);
    Ik_inc_list[i]=Ik;

    Ok_l3_list[i].resize(l3_ids.size());
    for(int j=0; j<l3_ids.size(); j++){
      if(Q_PEI[l3_ids[j]].size()==0){
        continue;
      }
      Eigen::VectorXcd Ok=Eigen::VectorXcd::Zero(num_expansion+1);
      compute_Ok(qi-Q_CNi(l3_ids[j]),(num_expansion+1),Ok);
      Ok_l3_list[i][j]=Ok;
    }
  }

}




void infinite::collect_expansions_g_vec(const int &num_expansion,
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
                      Eigen::MatrixXcd &MULg)
{
  Mg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
  Ug=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
  Lg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
  outgoing_from_source_g(sigma,leaf_cells,Q_PEI,Ik_out_list_list,num_expansion,Mg);
  outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mg);
  incoming_from_source_g(sigma,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug);
  incoming_from_outgoing_g(Q_PEI,Q_inters,Mg,num_expansion,Ok_inter_list_list,Lg);
  MULg = Lg+Ug;
  incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,MULg);
}


void infinite::collect_expansions_g(const int &num_expansion,
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
                      MatXcd_list &MULg_list)
{
  int nj = sigma.cols();
  Mg_list.resize(nj);
  Ug_list.resize(nj);
  Lg_list.resize(nj);
  MULg_list.resize(nj);


  for(int j=0; j<nj; j++){
    Eigen::MatrixXcd Mg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Ug=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Lg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

    outgoing_from_source_g(sigma.col(j),leaf_cells,Q_PEI,Ik_out_list_list,num_expansion,Mg);
    outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mg);
    incoming_from_source_g(sigma.col(j),Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug);
    incoming_from_outgoing_g(Q_PEI,Q_inters,Mg,num_expansion,Ok_inter_list_list,Lg);
    Mg_list[j] = Mg; Ug_list[j] = Ug; Lg_list[j] = Lg;
    Lg = Lg+Ug;
    incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,Lg);
    MULg_list[j] = Lg;
  }
}

void infinite::collect_expansions_f(const Eigen::VectorXcd &Ni,
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
                      MatXcd_list &MULf_list)
{
  int nj = mu.cols();
  Mf_list.resize(nj);
  Uf_list.resize(nj);
  Lf_list.resize(nj);
  MULf_list.resize(nj);

  for(int j=0; j<nj; j++){
    Eigen::MatrixXcd Mf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Uf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Lf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

    outgoing_from_source_f(mu.col(j),Ni,leaf_cells,Q_PEI,Ik_out_list_list,num_expansion,Mf);
    outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mf);
    // out_expansion_f_new(mu_s.col(j),Ni,leaf_cells,levels,Q_PEI,Ik_out_list_list,Ik_child_list,Q_CH,num_expansion,Mf);
    incoming_from_source_f(mu.col(j),Ni,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Uf);
    incoming_from_outgoing_f(Q_PEI,Q_inters,Mf,num_expansion,Ok_inter_list_list,Lf);
    Mf_list[j] = Mf; Uf_list[j] = Uf; Lf_list[j] = Lf;
    Lf = Lf+Uf;
    incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,Lf);
    MULf_list[j] = Lf;
  }
}




/////////
void infinite::eval_f(const int &num_expansion,
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
                    Eigen::MatrixXd &Wf_near)
{
  int nj = Mf_list.size();

  Wf_far = Eigen::MatrixXd::Zero(qI.size(),nj);
  Wf_near = Eigen::MatrixXd::Zero(qI.size(),nj);

  double PI_2 = 2*igl::PI;
  for(int i=0; i<qI.size(); i++){
    int cell_id = qI(i);
    std::vector<int> adj_ids = Q_adjs[cell_id];
    adj_ids.emplace_back(cell_id);
    for(int j=0; j<adj_ids.size(); j++){
      Eigen::VectorXi nI = nI_list[i][j];
      Eigen::MatrixXd mu_n=igl::slice(mu,nI,1);
      Wf_near.row(i) = Wf_near.row(i) + F_list[i][j]*mu_n;
    }

    Eigen::RowVectorXd f_far = Eigen::RowVectorXd::Zero(nj);
    Eigen::VectorXcd Ik = Ik_inc_list[i];
    for(int l=0; l<num_expansion; l++){
      for(int j=0; j<nj; j++){
        f_far(j) = f_far(j)+(Ik(l)*(-MULf_list[j](cell_id,l))).real();
      }
    }
    
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    for(int k=0; k<l3_ids.size(); k++){
      if(Q_PEI[l3_ids[k]].size()==0)
        continue;
      Eigen::VectorXcd Ok = Ok_l3_list[i][k];
      for(int l=0; l<num_expansion; l++){
        for(int j=0; j<nj; j++){
          f_far(j) = f_far(j)-(Ok(l+1)*Mf_list[j](l3_ids[k],l)).real();
        }
      }
    }
    Wf_far.row(i) = f_far/PI_2;
  }
}


void infinite::eval_g(const int &num_expansion,
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
                    Eigen::MatrixXd &Wg_near)
{
  int nj = Mg_list.size();
  Wg_far = Eigen::MatrixXd::Zero(qI.size(),nj);
  Wg_near = Eigen::MatrixXd::Zero(qI.size(),nj);

  double PI_2 = 2*igl::PI;
  for(int i=0; i<qI.size(); i++){
    int cell_id = qI(i);

    std::vector<int> adj_ids = Q_adjs[cell_id];
    adj_ids.emplace_back(cell_id);
    for(int j=0; j<adj_ids.size(); j++){
      Eigen::VectorXi nI = nI_list[i][j];
      Eigen::MatrixXd sigma_n=igl::slice(sigma,nI,1);
      Wg_near.row(i) = Wg_near.row(i) + G_list[i][j]*sigma_n;
    }

    Eigen::RowVectorXd g_far = Eigen::RowVectorXd::Zero(nj);
    Eigen::VectorXcd Ik = Ik_inc_list[i];
    for(int l=0; l<num_expansion; l++){
      for(int j=0; j<nj; j++){
        g_far(j) = g_far(j)+(Ik(l)*(MULg_list[j](cell_id,l))).real();
      }
    }
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    for(int k=0; k<l3_ids.size(); k++){
      if(Q_PEI[l3_ids[k]].size()==0)
        continue;
      Eigen::VectorXcd Ok = Ok_l3_list[i][k];
      for(int l=0; l<num_expansion; l++){
        for(int j=0; j<nj; j++){
          g_far(j) = g_far(j)+(Ok(l)*Mg_list[j](l3_ids[k],l)).real();
        }
      }
    }
    Wg_far.row(i) = g_far/PI_2;
  }
}


void infinite::eval_g_vec(const int &num_expansion,
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
                    Eigen::VectorXd &Wg_near)
{
  Wg_far = Eigen::VectorXd::Zero(qI.size());
  Wg_near = Eigen::VectorXd::Zero(qI.size());

  double PI_2 = 2*igl::PI;
  for(int i=0; i<qI.size(); i++){
    int cell_id = qI(i);

    std::vector<int> adj_ids = Q_adjs[cell_id];
    adj_ids.emplace_back(cell_id);
    for(int j=0; j<adj_ids.size(); j++){
      Eigen::VectorXi nI = nI_list[i][j];
      Eigen::VectorXd sigma_n = igl::slice(sigma,nI);
      Wg_near(i) = Wg_near(i) + G_list[i][j]*sigma_n;
    }

    double g_far = 0;
    Eigen::VectorXcd Ik = Ik_inc_list[i];
    for(int l=0; l<num_expansion; l++){
      g_far = g_far + (Ik(l)*(MULg(cell_id,l))).real();
    }
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    for(int k=0; k<l3_ids.size(); k++){
      if(Q_PEI[l3_ids[k]].size()==0)
        continue;
      Eigen::VectorXcd Ok = Ok_l3_list[i][k];
      for(int l=0; l<num_expansion; l++){
        g_far = g_far+(Ok(l)*Mg(l3_ids[k],l)).real();
      }
    }
    Wg_far(i) = g_far/PI_2;
  }
}




void infinite::pre_fmm(const Eigen::MatrixX2d &N,
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
                     VecXcd_list &Ik_inc_list)
{
  precompute_expansions_cell_dependent(N,L,num_expansion,Q_PEI,Q_LL,
    Q_PP,Q_CH,Q_LV,Q_CN,Q_big_seps,Q_inters,Ik_out_list_list,Ok_inter_list_list,
    Ik_child_list,Ok_inc_list_list,lw_list);
  precompute_expansions_query_dependent(N,L,Q,qI,sing_id_list,sing_t_list,
    num_expansion,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_CN,Q_adjs,Q_small_seps,leaf_cells,
    G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list);
}



void infinite::forward_fmm_f
                      (const Eigen::VectorXcd &Ni,
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
                      Eigen::MatrixXd &Wf)
{
  MatXcd_list Mf_list;
  MatXcd_list Uf_list;
  MatXcd_list Lf_list;
  MatXcd_list MULf_list;

  Eigen::MatrixXd Wf_far;
  Eigen::MatrixXd Wf_near;

  collect_expansions_f(Ni,num_expansion,mu,leaf_cells,levels,Q_PEI,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,Mf_list,Uf_list,Lf_list,MULf_list);

  eval_f(num_expansion,qI,Q_PEI,Q_adjs,Q_small_seps,F_list,nI_list,
    Ok_l3_list,Ik_inc_list,Mf_list,MULf_list,mu,Wf_far,Wf_near);

  Wf = Wf_far+Wf_near;
}


void infinite::forward_fmm_g
                      (const int &num_expansion,
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
                      Eigen::MatrixXd &Wg)
{
  MatXcd_list Mg_list;
  MatXcd_list Ug_list;
  MatXcd_list Lg_list;
  MatXcd_list MULg_list;

  Eigen::MatrixXd Wg_far;
  Eigen::MatrixXd Wg_near;

  collect_expansions_g(num_expansion,sigma,leaf_cells,levels,Q_PEI,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,Mg_list,Ug_list,Lg_list,MULg_list);

  eval_g(num_expansion,qI,Q_PEI,Q_adjs,Q_small_seps,G_list,nI_list,
    Ok_l3_list,Ik_inc_list,Mg_list,MULg_list,sigma,Wg_far,Wg_near);

  Wg = Wg_far+Wg_near;
}

void infinite::forward_fmm_g_vec
                      (const int &num_expansion,
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
                      Eigen::VectorXd &Wg)
{
  Eigen::MatrixXcd Mg;
  Eigen::MatrixXcd Ug;
  Eigen::MatrixXcd Lg;
  Eigen::MatrixXcd MULg;

  Eigen::VectorXd Wg_far;
  Eigen::VectorXd Wg_near;

  collect_expansions_g_vec(num_expansion,sigma,leaf_cells,levels,Q_PEI,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,Mg,Ug,Lg,MULg);


  eval_g_vec(num_expansion,qI,Q_PEI,Q_adjs,Q_small_seps,G_list,nI_list,
    Ok_l3_list,Ik_inc_list,Mg,MULg,sigma,Wg_far,Wg_near);

  Wg = Wg_far+Wg_near;

}




/////////////////////////////////////////////////////////////////////////

void infinite::update_expansions_cell_dependent
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
                     const std::vector<int> &update_big_seps,
                     std::vector<VecXcd_list> &Ik_out_list_list,
                     std::vector<VecXcd_list> &Ok_inter_list_list,
                     MatXcd_list &Ik_child_list,
                     std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                     VecXcd_list &lw_list)
{

  double PI_4 = 4*igl::PI;
  double PI_2 = 2*igl::PI;

  std::vector<std::vector<double> > Q_L(Q_LL.size());
  for(int i=0; i<Q_LL.size(); i++){
    Q_L[i].resize(Q_LL[i].size());
    for(int j=0; j<Q_LL[i].size(); j++){
      Q_L[i][j]=Q_LL[i][j](1)-Q_LL[i][j](0);
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);


  ///////////////////////////////////////////////

  for(int i=0; i<new_cells.size(); i++){
    Ok_inter_list_list.emplace_back(VecXcd_list());
    Ik_child_list.emplace_back(Eigen::MatrixXcd());
    Ik_out_list_list.emplace_back(VecXcd_list());
    Ok_inc_list_list.emplace_back(std::vector<VecXcd_list>());
    lw_list.emplace_back(Eigen::VectorXcd());
  }

  //update new interaction list
  for(int i=0; i<Q_inters_a.size(); i++){
    for(int j=0; j<Q_inters_a[i].size(); j++){
      Eigen::VectorXcd Ok = Eigen::VectorXcd::Zero(2*num_expansion);
      compute_Ok(Q_CNi[i]-Q_CNi[Q_inters_a[i][j]],2*num_expansion,Ok);
      Ok_inter_list_list[i].emplace_back(Ok);
    }
  }

  //update with new cells & sub cells
  // std::set<int> update_child_set;
  for(int i=0; i<sub_cells.size(); i++){
    int cell = sub_cells[i];
    if(Q_PEI[cell].size()==0)
      continue;
    if(Q_CH(cell,0)<0)
      continue;
    Eigen::MatrixXcd Ik(4,num_expansion);
    for(int ci=0; ci<4; ci++){
      compute_Ik(Q_CNi[Q_CH(cell,ci)]-Q_CNi[cell],num_expansion,Ik,ci);
    }
    Ik_child_list[cell]=Ik;
    // update_child_set.insert(cell);
  }

  for(int i=0; i<new_cells.size(); i++){
    int cell = new_cells[i];
    if(Q_PEI[cell].size()==0)
      continue;
    if(Q_CH(cell,0)<0)
      continue;
    Eigen::MatrixXcd Ik(4,num_expansion);
    for(int ci=0; ci<4; ci++){
      compute_Ik(Q_CNi[Q_CH(cell,ci)]-Q_CNi[cell],num_expansion,Ik,ci);
    }
    Ik_child_list[cell]=Ik;
    // update_child_set.insert(cell);
  }


  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }



  //collect cells that contain new_pids
  // std::set<int> update_cell_set;
  for(int i=0; i<update_leafs.size(); i++){
    int cell = update_leafs[i];   
    // bool update =false;
    // if(update_leaf_set.find(cell)==update_leaf_set.end())
    //   continue;
    std::complex<double> cni = Q_CNi(cell);
    Eigen::VectorXcd lw(Q_PEI[cell].size());
    VecXcd_list Ik_list(Q_PEI[cell].size());
    for(int ii=0; ii<Q_PEI[cell].size(); ii++){
      if(Q_L[cell][ii]<1e-15){
        lw(ii)=0;
        continue;
      }
      //fix: assumes tiny Q_L is already removed
      int pid = Q_PEI[cell][ii];
      Eigen::Matrix2d Pe = Q_PP[cell][ii];
      std::complex<double> za(Pe(0,0),Pe(0,1));
      std::complex<double> zb(Pe(1,0),Pe(1,1));
      const std::complex<double> &w = (zb-za)/std::norm(zb-za);
      lw(ii)=Q_L[cell][ii]*L(pid)*std::conj(w);
    }
    lw_list[cell]=lw;  
  }
  //need implementation of removing sub (leaf) cells
  

  //update with new (leaf) cells
  for(int i=0; i<update_leafs.size(); i++){
    int cell = update_leafs[i];   

    std::complex<double> cni = Q_CNi(cell);
    VecXcd_list Ik_list(Q_PEI[cell].size());
    for(int ii=0; ii<Q_PEI[cell].size(); ii++){
      if(Q_L[cell][ii]<1e-15){
        Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
        Ik_list[ii]=Ik;
        continue;
      }
      //fix: assumes tiny Q_L is already removed
      Eigen::Matrix2d Pe = Q_PP[cell][ii];
      std::complex<double> za(Pe(0,0),Pe(0,1));
      std::complex<double> zb(Pe(1,0),Pe(1,1));
      Eigen::VectorXcd Ik1 = Eigen::VectorXcd::Zero(num_expansion+1);
      Eigen::VectorXcd Ik2 = Eigen::VectorXcd::Zero(num_expansion+1);
      compute_Ik(zb-cni,num_expansion+1,Ik1);
      compute_Ik(za-cni,num_expansion+1,Ik2);
      Ik_list[ii] = lw_list[cell](ii)*(Ik1.tail(num_expansion)-Ik2.tail(num_expansion));
    }
    Ik_out_list_list[cell]=Ik_list;
    // Ik_out_list_list.emplace_back(Ik_list);
  }

  
  //update with l4 list
  // Ok_inc_list_list.resize(Q_big_seps.size());
  #pragma omp parallel for
  for(int i=0; i<update_big_seps.size(); i++){

    int cell = update_big_seps[i];
    const std::complex<double> &cni = Q_CNi(cell);
    // VecXcd_list wOk_list;
    Ok_inc_list_list[cell].resize(Q_big_seps[cell].size());

    for(int ii=0; ii<Q_big_seps[cell].size(); ii++){
      int l4_cell = Q_big_seps[cell][ii];
      Ok_inc_list_list[cell][ii].resize(Q_PEI[l4_cell].size());

      for(int jj=0; jj<Q_PEI[l4_cell].size(); jj++){
        // if(Q_L[l4_cell][jj]<1e-15){
        //   continue;
        // }
        int pid = Q_PEI[l4_cell][jj];
        Eigen::Matrix2d Pe = Q_PP[l4_cell][jj];
        std::complex<double> za(Pe(0,0),Pe(0,1));
        std::complex<double> zb(Pe(1,0),Pe(1,1));

        double l = Q_L[l4_cell][jj]*L(pid);

        Eigen::VectorXcd Ok1 = Eigen::VectorXcd::Zero(num_expansion-1);
        Eigen::VectorXcd Ok2 = Eigen::VectorXcd::Zero(num_expansion-1);
        compute_Ok_no_log(zb-cni,num_expansion-1,Ok1);
        compute_Ok_no_log(za-cni,num_expansion-1,Ok2);


        Eigen::VectorXcd Ok = Eigen::VectorXcd::Zero(num_expansion+1);
        Ok(0) = -green_line_integral(Pe,l,Q_CN.row(cell));
        Ok(1) = -lw_list[l4_cell](jj)*(log((zb-cni)/(za-cni)));
        Ok.tail(num_expansion-1) = lw_list[l4_cell](jj)*(Ok1-Ok2);
        Ok_inc_list_list[cell][ii][jj] = Ok;
      }
    }
  }
}





void infinite::update_expansions_query_dependent
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
                     VecXcd_list &Ik_inc_list)
{

  double PI_4 = 4*igl::PI;
  double PI_2 = 2*igl::PI;

  Eigen::VectorXcd Qi = mat2d_to_compvec(Q);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  std::vector<std::vector<double> > Q_L(Q_LL.size());
  for(int i=0; i<Q_LL.size(); i++){
    Q_L[i].resize(Q_LL[i].size());
    for(int j=0; j<Q_LL[i].size(); j++){
      Q_L[i][j]=Q_LL[i][j](1)-Q_LL[i][j](0);
    }
  }

  std::vector<int> sub_qids_vec(sub_qids.data(),sub_qids.data()+sub_qids.size());


  std::sort(sub_qids_vec.begin(),sub_qids_vec.end(),std::greater<int>());
  erase_indices(G_list,sub_qids_vec);
  erase_indices(F_list,sub_qids_vec);
  erase_indices(nI_list,sub_qids_vec);
  erase_indices(Ok_l3_list,sub_qids_vec);
  erase_indices(Ik_inc_list,sub_qids_vec);


  Eigen::VectorXi qI(Q.rows());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    for(int ci=0; ci<Q_QI[leaf_cells[i]].size(); ci++){
      qI(Q_QI[leaf_cells[i]][ci]) = leaf_cells[i];
    }
  }


  std::set<int> new_pid_set;
  for(int i=0; i<new_pids.size(); i++){
    new_pid_set.insert(new_pids(i));
  }

  std::set<int> new_cell_set;
  for(int i=0; i<new_cells.size(); i++){
    new_cell_set.insert(new_cells(i));
  }


  for(int i=0; i<(Q.rows()-new_qids.size()); i++){
    int cell_id = qI(i);

    std::vector<int> adj_ids = Q_adjs[cell_id];
    std::vector<int> near_ids = Q_PEI[cell_id];


    bool update_near=false;
    bool update_inc=false;
    bool update_l3=true;//false;
    if(new_cell_set.find(cell_id)!=new_cell_set.end()){
      update_near=true;
      update_inc=true;
      update_l3=true;
    }
    for(int j=0; j<adj_ids.size(); j++){
      near_ids.insert(near_ids.end(),Q_PEI[adj_ids[j]].begin(),Q_PEI[adj_ids[j]].end());
      if(new_cell_set.find(adj_ids[j])!=new_cell_set.end()){
        update_near=true;
      }
    }
    if(!update_near){
      for(int j=0; j<near_ids.size(); j++){
        if(new_pid_set.find(near_ids[j])!=new_pid_set.end()){
          update_near=true;
          break;
        }
      }
    }

    adj_ids.emplace_back(cell_id);

    nI_list[i].clear();
    nI_list[i].resize(adj_ids.size());
    for(int j=0; j<adj_ids.size(); j++){
      std::vector<int> near_ids = Q_PEI[adj_ids[j]];
      Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());
      nI_list[i][j]=nI;
    }
    // Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());
    // nI_list[i]=nI;
    if(update_near)
    {
      G_list[i].clear();
      F_list[i].clear();
      G_list[i].resize(adj_ids.size());
      F_list[i].resize(adj_ids.size());

      Eigen::VectorXi sI=sing_id_list[i];
      Eigen::VectorXd sT=sing_t_list[i];
      
      for(int j=0; j<adj_ids.size(); j++){

        if(nI_list[i][j].size()==0){
          continue;
        }
        Eigen::VectorXi nI = nI_list[i][j];

        Mat2d_list PEs = Q_PP[adj_ids[j]];
        std::vector<double> Ls = Q_L[adj_ids[j]];
        RowVec2d_list LL = Q_LL[adj_ids[j]];


        Eigen::MatrixX2d N_n=igl::slice(N,nI,1);
        Eigen::VectorXd L_n =igl::slice(L,nI);

        //ratio of length
        Eigen::VectorXd Lsv=Eigen::Map<Eigen::VectorXd>(Ls.data(),Ls.size());
        Lsv = Lsv.array()*L_n.array();

        Eigen::RowVectorXd G;
        Eigen::RowVectorXd F;
        green_line_integral(PEs,N_n,Lsv,Q.row(i),G,F);

        for(int ii=0; ii<sI.size(); ii++){
          for(int jj=0; jj<nI.size(); jj++){
            if(sI(ii)==nI(jj)){
              if((LL[jj](0)-1e-1)<sT(ii) && (LL[jj](1)+1e-1)>sT(ii)){
                double s_edge = Lsv(jj);
                double toe = (sT(ii)-LL[jj](0))/(LL[jj](1)-LL[jj](0))*s_edge;
                G(jj) = ((toe-s_edge)*log((toe-s_edge)*(toe-s_edge))
                            -toe*log(toe*toe)+2*s_edge)/PI_4;
                F(jj) = 0;
              }
            }
          }
        }
        G_list[i][j]=G;
        F_list[i][j]=F;
      }
    }

    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    if(!update_l3)
    {
      for(int j=0; j<l3_ids.size(); j++){
        if(new_cell_set.find(l3_ids[j])!=new_cell_set.end()){
          update_l3=true;
          break;
        }
        for(int jj=0; jj<Q_PEI[l3_ids[j]].size(); jj++){
          if(new_pid_set.find(Q_PEI[l3_ids[j]][jj])!=new_pid_set.end()){
            update_l3=true;
            break;
          }
        }
      }
    }
    if(update_inc)
    {
      std::complex<double> qi = Qi(i);
      std::complex<double> cni = Q_CNi(cell_id);
      Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
      compute_Ik(qi-cni, num_expansion, Ik);
      Ik_inc_list[i]=Ik;
    }
    if(update_l3)
    {
      // update_l3s.emplace_back(i);
      Ok_l3_list[i].clear();
      Ok_l3_list[i].resize(l3_ids.size());
      // Eigen::MatrixXcd Ok=Eigen::MatrixXcd::Zero(l3_ids.size(), num_expansion+1);
      for(int j=0; j<l3_ids.size(); j++){
        if(Q_PEI[l3_ids[j]].size()==0){
          continue;
        }
        Eigen::VectorXcd Ok=Eigen::VectorXcd::Zero(num_expansion+1);
        compute_Ok(Qi(i)-Q_CNi(l3_ids[j]),(num_expansion+1),Ok);
        Ok_l3_list[i][j]=Ok;
      }
    }
  }

  for(int i=0; i<new_qids.size(); i++){
    G_list.emplace_back(RowVecXd_list());
    F_list.emplace_back(RowVecXd_list());
    nI_list.emplace_back(VecXi_list());
    Ok_l3_list.emplace_back(VecXcd_list());
    Ik_inc_list.emplace_back(Eigen::VectorXcd());
  }

  //only run for new qids
  for(int i=0; i<new_qids.size(); i++){
    // update_nears.emplace_back(new_qids(i));
    // update_l3s.emplace_back(new_qids(i));

    //fix: use data from adap quadtree
    int cell_id = qI(new_qids(i));
    // int cell_id = query_cell(Q_CH,Q_CN,Q_W,Q.row(new_qids(i)));
    std::complex<double> qi = Qi(new_qids(i));
    std::vector<int> adj_ids = Q_adjs[cell_id];
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];

    adj_ids.emplace_back(cell_id);

    G_list[new_qids(i)].resize(adj_ids.size());
    F_list[new_qids(i)].resize(adj_ids.size());
    nI_list[new_qids(i)].resize(adj_ids.size());

    Eigen::VectorXi sI=sing_id_list[new_qids(i)];
    Eigen::VectorXd sT=sing_t_list[new_qids(i)];

    for(int j=0; j<adj_ids.size(); j++){
      std::vector<int> near_ids = Q_PEI[adj_ids[j]];
      if(near_ids.size()==0)
        continue;
      Mat2d_list PEs = Q_PP[adj_ids[j]];
      std::vector<double> Ls = Q_L[adj_ids[j]];
      RowVec2d_list LL = Q_LL[adj_ids[j]];
      Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());
      nI_list[new_qids(i)][j]=nI;

      Eigen::MatrixX2d N_n=igl::slice(N,nI,1);
      Eigen::VectorXd L_n =igl::slice(L,nI);

      //ratio of length
      Eigen::VectorXd Lsv=Eigen::Map<Eigen::VectorXd>(Ls.data(),Ls.size());
      Lsv = Lsv.array()*L_n.array();


      Eigen::RowVectorXd G;
      Eigen::RowVectorXd F;
      green_line_integral(PEs,N_n,Lsv,Q.row(new_qids(i)),G,F);

      for(int ii=0; ii<sI.size(); ii++){
        for(int jj=0; jj<nI.size(); jj++){
          if(sI(ii)==nI(jj)){
            if((LL[jj](0)-1e-1)<sT(ii) && (LL[jj](1)+1e-1)>sT(ii)){
              double s_edge = Lsv(jj);
              double toe = (sT(ii)-LL[jj](0))/(LL[jj](1)-LL[jj](0))*s_edge;
              G(jj) = ((toe-s_edge)*log((toe-s_edge)*(toe-s_edge))
                          -toe*log(toe*toe)+2*s_edge)/PI_4;
              F(jj) = 0;
            }
          }
        }
      }
      G_list[new_qids(i)][j]=G;
      F_list[new_qids(i)][j]=F;
    }

    std::complex<double> cni = Q_CNi(cell_id);

    Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
    compute_Ik(qi-cni, num_expansion, Ik);
    Ik_inc_list[new_qids(i)]=Ik;


    Ok_l3_list[new_qids(i)].resize(l3_ids.size());
    for(int j=0; j<l3_ids.size(); j++){
      if(Q_PEI[l3_ids[j]].size()==0){
        continue;
      }
      Eigen::VectorXcd Ok=Eigen::VectorXcd::Zero(num_expansion+1);
      compute_Ok(qi-Q_CNi(l3_ids[j]),(num_expansion+1),Ok);
      Ok_l3_list[new_qids(i)][j]=Ok;
    }

  }
}




void infinite::update_collecting_expansions_g_vec
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
                      Eigen::MatrixXcd &MULg)
{

  Mg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
  Ug=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
  Lg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

  update_outgoing_from_source_g(update_pids,sigma,leaf_cells,Q_PEI,Ik_out_list_list,num_expansion,Mg);
  outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mg);
  update_incoming_from_source_g(update_pids,sigma,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug);

  incoming_from_outgoing_g(Q_PEI,Q_inters,Mg,num_expansion,Ok_inter_list_list,Lg);
  MULg = Lg+Ug;
  incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,MULg);  
}



void infinite::update_collecting_expansions_g
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
                      MatXcd_list &MULg_list)
{
  int nj = sigma.cols();
  
  Mg_list.resize(nj);
  Ug_list.resize(nj);
  Lg_list.resize(nj);
  MULg_list.resize(nj);

  for(int j=0; j<nj; j++){
    Eigen::MatrixXcd Mg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Ug=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Lg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

    update_outgoing_from_source_g(update_pids,sigma.col(j),leaf_cells,Q_PEI,Ik_out_list_list,num_expansion,Mg);
    outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mg);
    update_incoming_from_source_g(update_pids,sigma.col(j),Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug);

    incoming_from_outgoing_g(Q_PEI,Q_inters,Mg,num_expansion,Ok_inter_list_list,Lg);
    Mg_list[j] = Mg; Ug_list[j] = Ug; Lg_list[j] = Lg;
    Lg = Lg+Ug;
    incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,Lg);
    MULg_list[j] = Lg;
  }
}

void infinite::update_collecting_expansions_f
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
                      MatXcd_list &MULf_list)
{
  int nj = mu.cols();
  
  Mf_list.resize(nj);
  Uf_list.resize(nj);
  Lf_list.resize(nj);
  MULf_list.resize(nj);

  for(int j=0; j<nj; j++){
    Eigen::MatrixXcd Mf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Uf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Eigen::MatrixXcd Lf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

    update_outgoing_from_source_f(update_pids,mu.col(j),Ni,leaf_cells,Q_PEI,Ik_out_list_list,num_expansion,Mf);
    outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mf);
    update_incoming_from_sourc_f(update_pids,mu.col(j),Ni,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Uf);

    incoming_from_outgoing_f(Q_PEI,Q_inters,Mf,num_expansion,Ok_inter_list_list,Lf);
    Mf_list[j] = Mf; Uf_list[j] = Uf; Lf_list[j] = Lf;
    Lf = Lf+Uf;
    incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,Lf);
    MULf_list[j] = Lf;
  }
}




void infinite::update_eval_g_vec(const int &num_expansion,
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
                    Eigen::VectorXd &Wg_near)
{

  Wg_far = Eigen::VectorXd::Zero(sol_qids.size());
  Wg_near = Eigen::VectorXd::Zero(sol_qids.size());

  double PI_2 = 2*igl::PI;
  for(int i=0; i<sol_qids.size(); i++){
    int qid = sol_qids(i);
    int cell_id = qI(qid);
    std::vector<int> adj_ids = Q_adjs[cell_id];
    adj_ids.emplace_back(cell_id);

    for(int j=0; j<adj_ids.size(); j++){
      Eigen::VectorXi nI = nI_list[qid][j];
      std::vector<int> valid_pids;
      std::vector<int> valid_gids;
      for(int k=0; k<nI.size(); k++){
        if(!update_pids[nI(k)])
          continue;
        valid_pids.emplace_back(nI(k));
        valid_gids.emplace_back(k);
      }
      Eigen::VectorXi vPI=Eigen::Map<Eigen::VectorXi>(valid_pids.data(),valid_pids.size());
      Eigen::VectorXi vGI=Eigen::Map<Eigen::VectorXi>(valid_gids.data(),valid_gids.size());

      Eigen::VectorXd sigma_n=igl::slice(sigma,vPI);
      Eigen::RowVectorXd G_n = igl::slice(G_list[qid][j],vGI,2);
      Wg_near(i) = Wg_near(i) + G_n*sigma_n;
    }
    
    double g_far = 0;
    Eigen::VectorXcd Ik = Ik_inc_list[qid];
    for(int l=0; l<num_expansion; l++){
      g_far = g_far+(Ik(l)*(MULg(cell_id,l))).real();
    }
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    for(int k=0; k<l3_ids.size(); k++){
      if(Q_PEI[l3_ids[k]].size()==0)
        continue;
      Eigen::VectorXcd Ok = Ok_l3_list[qid][k];
      for(int l=0; l<num_expansion; l++){
        g_far = g_far+(Ok(l)*Mg(l3_ids[k],l)).real();
      }
    }
    Wg_far(i) = g_far/PI_2;
  }
}

void infinite::update_eval_g(const int &num_expansion,
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
                    Eigen::MatrixXd &Wg_near)
{
  int nj = Mg_list.size();

  Wg_far = Eigen::MatrixXd::Zero(sol_qids.size(),nj);
  Wg_near = Eigen::MatrixXd::Zero(sol_qids.size(),nj);

  double PI_2 = 2*igl::PI;
  for(int i=0; i<sol_qids.size(); i++){
    int qid = sol_qids(i);
    int cell_id = qI(qid);
    std::vector<int> adj_ids = Q_adjs[cell_id];
    adj_ids.emplace_back(cell_id);

    for(int j=0; j<adj_ids.size(); j++){
      Eigen::VectorXi nI = nI_list[qid][j];

      std::vector<int> valid_pids;
      std::vector<int> valid_gids;
      for(int k=0; k<nI.size(); k++){
        if(!update_pids[nI(k)])
          continue;
        valid_pids.emplace_back(nI(k));
        valid_gids.emplace_back(k);
      }
      Eigen::VectorXi vPI=Eigen::Map<Eigen::VectorXi>(valid_pids.data(),valid_pids.size());
      Eigen::VectorXi vGI=Eigen::Map<Eigen::VectorXi>(valid_gids.data(),valid_gids.size());

      Eigen::MatrixXd sigma_n=igl::slice(sigma,vPI,1);
      Eigen::RowVectorXd G_n = igl::slice(G_list[qid][j],vGI,2);
      Wg_near.row(i) = Wg_near.row(i) + G_n*sigma_n;
    }
    
    Eigen::RowVectorXd g_far = Eigen::RowVectorXd::Zero(nj);
    Eigen::VectorXcd Ik = Ik_inc_list[qid];
    for(int l=0; l<num_expansion; l++){
      for(int j=0; j<nj; j++){
        g_far(j) = g_far(j)+(Ik(l)*(MULg_list[j](cell_id,l))).real();
      }
    }

    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    for(int k=0; k<l3_ids.size(); k++){
      if(Q_PEI[l3_ids[k]].size()==0)
        continue;
      Eigen::VectorXcd Ok = Ok_l3_list[qid][k];
      for(int l=0; l<num_expansion; l++){
        for(int j=0; j<nj; j++){
          g_far(j) = g_far(j)+(Ok(l)*Mg_list[j](l3_ids[k],l)).real();
        }
      }
    }
    Wg_far.row(i) = g_far/PI_2;
  }
}


void infinite::update_eval_f(const int &num_expansion,
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
                    Eigen::MatrixXd &Wf_near)
{
  int nj = Mf_list.size();

  Wf_far = Eigen::MatrixXd::Zero(sol_qids.size(),nj);
  Wf_near = Eigen::MatrixXd::Zero(sol_qids.size(),nj);

  double PI_2 = 2*igl::PI;
  for(int i=0; i<sol_qids.size(); i++){
    int qid = sol_qids(i);
    int cell_id = qI(qid);
    std::vector<int> adj_ids = Q_adjs[cell_id];
    adj_ids.emplace_back(cell_id);

    for(int j=0; j<adj_ids.size(); j++){
      Eigen::VectorXi nI = nI_list[qid][j];
      std::vector<int> valid_pids;
      std::vector<int> valid_gids;
      for(int k=0; k<nI.size(); k++){
        if(!update_pids[nI(k)])
          continue;
        valid_pids.emplace_back(nI(k));
        valid_gids.emplace_back(k);
      }
      Eigen::VectorXi vPI=Eigen::Map<Eigen::VectorXi>(valid_pids.data(),valid_pids.size());
      Eigen::VectorXi vGI=Eigen::Map<Eigen::VectorXi>(valid_gids.data(),valid_gids.size());

      Eigen::MatrixXd mu_n=igl::slice(mu,vPI,1);
      Eigen::RowVectorXd F_n = igl::slice(F_list[qid][j],vGI,2);
      Wf_near.row(i) = Wf_near.row(i) + F_n*mu_n;
    }
    
    Eigen::RowVectorXd f_far = Eigen::RowVectorXd::Zero(nj);
    Eigen::VectorXcd Ik = Ik_inc_list[qid];
    for(int l=0; l<num_expansion; l++){
      for(int j=0; j<nj; j++){
        f_far(j) = f_far(j)+(Ik(l)*(-MULf_list[j](cell_id,l))).real();
      }
    }

    const std::vector<int> &l3_ids = Q_small_seps[cell_id];
    for(int k=0; k<l3_ids.size(); k++){
      if(Q_PEI[l3_ids[k]].size()==0)
        continue;
      Eigen::VectorXcd Ok = Ok_l3_list[qid][k];
      for(int l=0; l<num_expansion; l++){
        for(int j=0; j<nj; j++){
          f_far(j) = f_far(j)-(Ok(l+1)*Mf_list[j](l3_ids[k],l)).real();
        }
      }
    }
    Wf_far.row(i) = f_far/PI_2;
  }
}


void infinite::update_pre_fmm_hybrid
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
                     const std::vector<int> &update_big_seps,
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
                     VecXcd_list &Ik_inc_list)
{
  update_expansions_cell_dependent(N,L,num_expansion,Q_PEI,Q_LL,
    Q_PP,Q_CH,Q_LV,Q_CN,sub_cells,new_cells,Q_big_seps,Q_inters_a,update_leafs,
    update_big_seps,Ik_out_list_list,Ok_inter_list_list,
    Ik_child_list,Ok_inc_list_list,lw_list);

  update_expansions_query_dependent(N,L,Q,sing_id_list,sing_t_list,
    num_expansion,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_CN,new_cells,Q_adjs,Q_small_seps,
    leaf_cells,new_pids,sub_qids,new_qids,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list);
}


void infinite::update_forward_fmm_f(
                    const Eigen::VectorXcd &Ni,
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
                    Eigen::MatrixXd &Wf)
{
  MatXcd_list Mf_list;
  MatXcd_list Uf_list;
  MatXcd_list Lf_list;
  MatXcd_list MULf_list;

  Eigen::MatrixXd Wf_far;
  Eigen::MatrixXd Wf_near;

  update_collecting_expansions_f(Ni,num_expansion,mu,leaf_cells,
    levels,Q_PEI,Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,update_pids,Mf_list,Uf_list,Lf_list,MULf_list);

  update_eval_f(num_expansion,qI,sol_qids,Q_PEI,Q_adjs,Q_small_seps,F_list,
    nI_list,Ok_l3_list,Ik_inc_list,Mf_list,MULf_list,mu,update_pids,
    Wf_far,Wf_near);

  Wf = Wf_far + Wf_near;
  
}

void infinite::update_forward_fmm_g(
                    const int &num_expansion,
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
                    Eigen::MatrixXd &Wg)
{
  MatXcd_list Mg_list;
  MatXcd_list Ug_list;
  MatXcd_list Lg_list;
  MatXcd_list MULg_list;

  Eigen::MatrixXd Wg_far;
  Eigen::MatrixXd Wg_near;

  update_collecting_expansions_g(num_expansion,sigma,leaf_cells,
    levels,Q_PEI,Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,update_pids,Mg_list,Ug_list,Lg_list,MULg_list);

  update_eval_g(num_expansion,qI,sol_qids,Q_PEI,Q_adjs,Q_small_seps,G_list,
    nI_list,Ok_l3_list,Ik_inc_list,Mg_list,MULg_list,sigma,update_pids,
    Wg_far,Wg_near);

  Wg = Wg_far + Wg_near;
  
}


void infinite::update_forward_fmm_g_vec(
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
                    Eigen::VectorXd &Wg)
{
  Eigen::MatrixXcd Mg;
  Eigen::MatrixXcd Ug;
  Eigen::MatrixXcd Lg;
  Eigen::MatrixXcd MULg;

  Eigen::VectorXd Wg_far;
  Eigen::VectorXd Wg_near;

  infinite::update_collecting_expansions_g_vec(num_expansion,sigma,leaf_cells,levels,Q_PEI,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,update_pids,
    Mg,Ug,Lg,MULg);

  infinite::update_eval_g_vec(num_expansion,qI,sol_qids,Q_PEI,Q_adjs,Q_small_seps,G_list,nI_list,
    Ok_l3_list,Ik_inc_list,Mg,MULg,sigma,update_pids,Wg_far,Wg_near);

  Wg = Wg_far + Wg_near;
}