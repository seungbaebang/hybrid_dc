#include "eval_fmm.h"



void infinite::eval_fmm_integral_noclip(
                  const Eigen::MatrixXd &sigma,
                  const Eigen::MatrixXd &mu,
                  const int &num_expansion,
                  const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const std::vector<std::vector<int > > &Q_PI,
                  const std::vector<std::vector<int > > &Q_QI,
                  const std::vector<std::vector<int> > &levels,
                  const Eigen::MatrixXi &Q_CH,
                  const Eigen::VectorXi &Q_LV,
                  const Eigen::MatrixXd &Q_CN, 
                  const std::vector<std::vector<int> > &Q_adjs,
                  const std::vector<std::vector<int> > &Q_small_seps,
                  const std::vector<std::vector<int> > &Q_inters,
                  const std::vector<std::vector<int> > &Q_big_seps,
                  const std::vector<int> &leaf_cells,
                  Eigen::MatrixXd& W)
{

  std::vector<VecXcd_list> Ik_out_list_list;
  std::vector<VecXcd_list> Ok_inter_list_list;
  MatXcd_list Ik_child_list;
  std::vector<std::vector<VecXcd_list> > Ok_inc_list_list;
  // VecXcd_list lw_list;

  precompute_expansions_cell_dependent_noclip(P,E,N,L,num_expansion,Q_PI,
    Q_CH,Q_LV,Q_CN,Q_inters,Q_big_seps,Ik_out_list_list,Ok_inter_list_list,
    Ik_child_list,Ok_inc_list_list);


  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  Eigen::VectorXcd Qi = mat2d_to_compvec(Q);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  int nj = sigma.cols();



  MatXcd_list Lg_list(nj);
  MatXcd_list Lf_list(nj);
  MatXcd_list Mg_list(nj);
  MatXcd_list Mf_list(nj);


  // #pragma omp parallel for
  for(int j=0; j<nj; j++){
    Eigen::MatrixXcd Mg,Mf,Ug,Uf,Lg,Lf;

    Mg=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Ug=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Lg=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Mf=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Uf=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Lf=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);

    outgoing_from_source(sigma.col(j),mu.col(j),Ni,leaf_cells,Q_PI,
                          Ik_out_list_list,num_expansion,Mg,Mf);
    outgoing_from_outgoing(levels,Q_PI,Ik_child_list,Q_CH,num_expansion,Mg,Mf);

    incoming_from_source(sigma.col(j),mu.col(j),Ni,Ok_inc_list_list,Q_big_seps,Q_PI,num_expansion,Ug,Uf);

    incoming_from_outgoing(Q_PI,Q_inters,Mg,Mf,num_expansion,Ok_inter_list_list,Lg,Lf);
    Lg = Lg+Ug; Lf = Lf+Uf;
    incoming_from_incoming(Q_PI,levels,Q_CH,num_expansion,Ik_child_list,Lg,Lf);

    Mg_list[j] = Mg;  Mf_list[j] = Mf;
    Lg_list[j] = Lg;  Lf_list[j] = Lf;

  }
  std::cout<<"4"<<std::endl;

  W = Eigen::MatrixXd::Zero(Q.rows(),nj);

  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int cell_id = leaf_cells[i];
    std::complex<double> cni = Q_CNi(cell_id);
    const std::vector<int> &adj_ids = Q_adjs[cell_id];
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];

    std::vector<int> near_ids = Q_PI[cell_id];
    for(int j=0; j<adj_ids.size(); j++){
      near_ids.insert(near_ids.end(),Q_PI[adj_ids[j]].begin(),Q_PI[adj_ids[j]].end());
    }
    Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());

    Eigen::MatrixXd sigma_n=igl::slice(sigma,nI,1);
    Eigen::MatrixXd mu_n=igl::slice(mu,nI,1);
    Eigen::MatrixX2d N_n=igl::slice(N,nI,1);
    Eigen::VectorXd L_n =igl::slice(L,nI);

    
    for(int ci=0; ci<Q_QI[cell_id].size(); ci++){
      int qid = Q_QI[cell_id][ci];
      std::complex<double> qi = Qi(qid);

      if(near_ids.size()>0){
        Eigen::RowVectorXd G;
        Eigen::RowVectorXd F;
        // green_line_integral(PEs,N_n,Lsv,Q.row(qid),G,F);

        green_line_integral(P,E,N,L,near_ids,Q.row(qid),G,F);
        W.row(qid) = G*sigma_n + F*mu_n;
      }

      Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
      compute_Ik(qi-cni, num_expansion, Ik);

      Eigen::MatrixXcd Ok=Eigen::MatrixXcd::Zero(l3_ids.size(), num_expansion+1);
      for(int j=0; j<l3_ids.size(); j++){
        if(Q_PI[l3_ids[j]].size()==0){
          continue;
        }
        compute_Ok(qi-Q_CNi(l3_ids[j]),(num_expansion+1),Ok,j);
      }

      //#pragma omp parallel for
      for(int j=0; j<nj; j++){
        Eigen::MatrixXcd& Mg = Mg_list[j];  Eigen::MatrixXcd& Mf = Mf_list[j];
        Eigen::MatrixXcd& Lg = Lg_list[j];  Eigen::MatrixXcd& Lf = Lf_list[j];
        for(int l=0; l<num_expansion; l++){
          for(int k=0; k<l3_ids.size(); k++){
            if(Q_PI[l3_ids[k]].size()==0)
              continue;
            W(qid,j)=W(qid,j)+(Ok(k,l)*Mg(l3_ids[k],l)-Ok(k,l+1)*Mf(l3_ids[k],l)).real()/(2*igl::PI);
          }
          W(qid,j)=W(qid,j)+(Ik(l)*(Lg(cell_id,l)-Lf(cell_id,l))).real()/(2*igl::PI);
        }
      }
    }
  }
}

void infinite::eval_fmm_integral_noclip(const Eigen::MatrixX2d &P,
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
                       Eigen::MatrixXd &W)
{
  Eigen::MatrixX2d PQ(P.rows()+Q.rows(),P.cols());
  PQ<<P,Q;

  Eigen::RowVector2d minQ = PQ.colwise().minCoeff();
  Eigen::RowVector2d maxQ = PQ.colwise().maxCoeff();

  std::vector<std::vector<int > > Q_PI, Q_QI;
  Eigen::MatrixXi Q_CH;
  Eigen::VectorXi Q_PA, Q_LV;
  Eigen::MatrixXd Q_CN;
  Eigen::VectorXd Q_W;
  std::vector<std::vector<int> > Q_adjs, Q_small_seps, Q_big_seps, Q_inters;
  quadtree(C,Q,minQ,maxQ,max_depth,min_pnt_num,
                Q_PI,Q_QI,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);

  compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_small_seps,Q_big_seps,Q_inters);

  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  // std::cout<<"max level: "<<Q_LV.maxCoeff()<<std::endl;

  eval_fmm_integral_noclip(sigma,mu,num_expansion,P,E,N,L,Q,Q_PI,Q_QI,levels,
          Q_CH,Q_LV,Q_CN,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,W);

}



void infinite::eval_fmm_integral(
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
                  Eigen::MatrixXd& W)
{
  std::vector<std::vector<double> > Q_L(Q_LL.size());
  for(int i=0; i<Q_LL.size(); i++){
    Q_L[i].resize(Q_LL[i].size());
    for(int j=0; j<Q_LL[i].size(); j++){
      Q_L[i][j]=Q_LL[i][j](1)-Q_LL[i][j](0);
    }
  }
  std::vector<VecXcd_list> Ik_out_list_list;
  std::vector<VecXcd_list> Ok_inter_list_list;
  MatXcd_list Ik_child_list;
  std::vector<std::vector<VecXcd_list> > Ok_inc_list_list;
  VecXcd_list lw_list;

  precompute_expansions_cell_dependent(N,L,num_expansion,Q_PEI,Q_LL,
    Q_PP,Q_CH,Q_LV,Q_CN,Q_big_seps,Q_inters,Ik_out_list_list,Ok_inter_list_list,
    Ik_child_list,Ok_inc_list_list,lw_list);


  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  Eigen::VectorXcd Qi = mat2d_to_compvec(Q);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  int nj = sigma.cols();



  MatXcd_list Lg_list(nj);
  MatXcd_list Lf_list(nj);
  MatXcd_list Mg_list(nj);
  MatXcd_list Mf_list(nj);


  // #pragma omp parallel for
  for(int j=0; j<nj; j++){
    Eigen::MatrixXcd Mg,Mf,Ug,Uf,Lg,Lf;

    Mg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Ug=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Lg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Mf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Uf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
    Lf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

    outgoing_from_source(sigma.col(j),mu.col(j),Ni,leaf_cells,Q_PEI,
                          Ik_out_list_list,num_expansion,Mg,Mf);
    outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mg,Mf);

    incoming_from_source(sigma.col(j),mu.col(j),Ni,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug,Uf);

    incoming_from_outgoing(Q_PEI,Q_inters,Mg,Mf,num_expansion,Ok_inter_list_list,Lg,Lf);
    Lg = Lg+Ug; Lf = Lf+Uf;
    incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,Lg,Lf);


    Mg_list[j] = Mg;  Mf_list[j] = Mf;
    Lg_list[j] = Lg;  Lf_list[j] = Lf;

  }

  W = Eigen::MatrixXd::Zero(Q.rows(),nj);

  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int cell_id = leaf_cells[i];
    std::complex<double> cni = Q_CNi(cell_id);
    const std::vector<int> &adj_ids = Q_adjs[cell_id];
    const std::vector<int> &l3_ids = Q_small_seps[cell_id];

    std::vector<int> near_ids = Q_PEI[cell_id];
    Mat2d_list PEs = Q_PP[cell_id];
    std::vector<double> Ls = Q_L[cell_id];
    for(int j=0; j<adj_ids.size(); j++){
      near_ids.insert(near_ids.end(),Q_PEI[adj_ids[j]].begin(),Q_PEI[adj_ids[j]].end());
      PEs.insert(PEs.end(),Q_PP[adj_ids[j]].begin(),Q_PP[adj_ids[j]].end());
      Ls.insert(Ls.end(),Q_L[adj_ids[j]].begin(),Q_L[adj_ids[j]].end());
    }
    Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());

    Eigen::MatrixXd sigma_n=igl::slice(sigma,nI,1);
    Eigen::MatrixXd mu_n=igl::slice(mu,nI,1);
    Eigen::MatrixX2d N_n=igl::slice(N,nI,1);
    Eigen::VectorXd L_n =igl::slice(L,nI);

    Eigen::VectorXd Lsv=Eigen::Map<Eigen::VectorXd>(Ls.data(),Ls.size());
    Lsv = Lsv.array()*L_n.array();
    
    for(int ci=0; ci<Q_QI[cell_id].size(); ci++){
      int qid = Q_QI[cell_id][ci];
      std::complex<double> qi = Qi(qid);

      if(near_ids.size()>0){
        Eigen::RowVectorXd G;
        Eigen::RowVectorXd F;
        green_line_integral(PEs,N_n,Lsv,Q.row(qid),G,F);
        W.row(qid) = G*sigma_n + F*mu_n;
      }

      Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
      compute_Ik(qi-cni, num_expansion, Ik);

      Eigen::MatrixXcd Ok=Eigen::MatrixXcd::Zero(l3_ids.size(), num_expansion+1);
      for(int j=0; j<l3_ids.size(); j++){
        if(Q_PEI[l3_ids[j]].size()==0){
          continue;
        }
        compute_Ok(qi-Q_CNi(l3_ids[j]),(num_expansion+1),Ok,j);
      }

      //#pragma omp parallel for
      for(int j=0; j<nj; j++){
        Eigen::MatrixXcd& Mg = Mg_list[j];  Eigen::MatrixXcd& Mf = Mf_list[j];
        Eigen::MatrixXcd& Lg = Lg_list[j];  Eigen::MatrixXcd& Lf = Lf_list[j];
        for(int l=0; l<num_expansion; l++){
          for(int k=0; k<l3_ids.size(); k++){
            if(Q_PEI[l3_ids[k]].size()==0)
              continue;
            W(qid,j)=W(qid,j)+(Ok(k,l)*Mg(l3_ids[k],l)-Ok(k,l+1)*Mf(l3_ids[k],l)).real()/(2*igl::PI);
          }
          W(qid,j)=W(qid,j)+(Ik(l)*(Lg(cell_id,l)-Lf(cell_id,l))).real()/(2*igl::PI);
        }
      }
    }
  }
}

void infinite::eval_fmm_integral(const Eigen::MatrixX2d &P,
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
                       Eigen::MatrixXd &W)
{
  Eigen::MatrixX2d PQ(P.rows()+Q.rows(),P.cols());
  PQ<<P,Q;

  Eigen::RowVector2d minQ = PQ.colwise().minCoeff();
  Eigen::RowVector2d maxQ = PQ.colwise().maxCoeff();

  std::vector<std::vector<int > > Q_PI, Q_PEI, Q_QI;
  std::vector<std::vector<double> > Q_L;
  Eigen::MatrixXi Q_CH;
  Eigen::VectorXi Q_PA, Q_LV;
  Eigen::MatrixXd Q_CN;
  Eigen::VectorXd Q_W;
  std::vector<std::vector<int> > Q_adjs, Q_small_seps, Q_big_seps, Q_inters;
  Mat2d_list_list Q_PP;
  RowVec2d_list_list Q_LL;

  quadtree(P,E,C,Q,minQ,maxQ,max_depth,min_pnt_num,
                Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);
  compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_small_seps,Q_big_seps,Q_inters);

  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  // std::cout<<"max level: "<<Q_LV.maxCoeff()<<std::endl;

  eval_fmm_integral(sigma,mu,num_expansion,N,L,Q,Q_PEI,Q_QI,levels,Q_LL,
          Q_PP,Q_CH,Q_LV,Q_CN,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,W);
}


// void infinite::eval_fmm_integral(
//                   const Eigen::MatrixXd &sigma,
//                   const Eigen::MatrixXd &mu,
//                   const int &num_expansion,
//                   const Eigen::MatrixX2d &N,
//                   const Eigen::VectorXd &L,
//                   const Eigen::MatrixX2d &Q,
//                   const std::vector<std::vector<int > > &Q_PEI,
//                   const std::vector<std::vector<int > > &Q_QI,
//                   const std::vector<std::vector<int> > &levels,
//                   const RowVec2d_list_list &Q_LL,
//                   const Mat2d_list_list &Q_PP,
//                   const Eigen::MatrixXi &Q_CH,
//                   const Eigen::VectorXi &Q_LV,
//                   const Eigen::MatrixXd &Q_CN, 
//                   const std::vector<std::vector<int> > &Q_adjs,
//                   const std::vector<std::vector<int> > &Q_small_seps,
//                   const std::vector<std::vector<int> > &Q_inters,
//                   const std::vector<std::vector<int> > &Q_big_seps,
//                   const std::vector<int> &leaf_cells,
//                   Eigen::MatrixXd& W)
// {
//   std::vector<std::vector<double> > Q_L(Q_LL.size());
//   for(int i=0; i<Q_LL.size(); i++){
//     Q_L[i].resize(Q_LL[i].size());
//     for(int j=0; j<Q_LL[i].size(); j++){
//       Q_L[i][j]=Q_LL[i][j](1)-Q_LL[i][j](0);
//     }
//   }
//   std::vector<VecXcd_list> Ik_out_list_list;
//   std::vector<VecXcd_list> Ok_inter_list_list;
//   MatXcd_list Ik_child_list;
//   std::vector<std::vector<VecXcd_list> > Ok_inc_list_list;
//   VecXcd_list lw_list;

//   precompute_expansions_cell_dependent(N,L,num_expansion,Q_PEI,Q_LL,
//     Q_PP,Q_CH,Q_LV,Q_CN,Q_big_seps,Q_inters,Ik_out_list_list,Ok_inter_list_list,
//     Ik_child_list,Ok_inc_list_list,lw_list);


//   Eigen::VectorXcd Ni = mat2d_to_compvec(N);
//   Eigen::VectorXcd Qi = mat2d_to_compvec(Q);
//   Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

//   int nj = sigma.cols();



//   MatXcd_list Lg_list(nj);
//   MatXcd_list Lf_list(nj);
//   MatXcd_list Mg_list(nj);
//   MatXcd_list Mf_list(nj);


//   // #pragma omp parallel for
//   for(int j=0; j<nj; j++){
//     Eigen::MatrixXcd Mg,Mf,Ug,Uf,Lg,Lf;

//     Mg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Ug=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Lg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Mf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Uf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Lf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

//     outgoing_from_source(sigma.col(j),mu.col(j),Ni,leaf_cells,Q_PEI,
//                           Ik_out_list_list,num_expansion,Mg,Mf);
//     outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mg,Mf);

//     incoming_from_source(sigma.col(j),mu.col(j),Ni,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug,Uf);

//     incoming_from_outgoing(Q_PEI,Q_inters,Mg,Mf,num_expansion,Ok_inter_list_list,Lg,Lf);
//     Lg = Lg+Ug; Lf = Lf+Uf;
//     incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,Lg,Lf);


//     Mg_list[j] = Mg;  Mf_list[j] = Mf;
//     Lg_list[j] = Lg;  Lf_list[j] = Lf;

//   }

//   W = Eigen::MatrixXd::Zero(Q.rows(),nj);

//   #pragma omp parallel for
//   for(int i=0; i<leaf_cells.size(); i++){
//     int cell_id = leaf_cells[i];
//     std::complex<double> cni = Q_CNi(cell_id);
//     const std::vector<int> &adj_ids = Q_adjs[cell_id];
//     const std::vector<int> &l3_ids = Q_small_seps[cell_id];

//     std::vector<int> near_ids = Q_PEI[cell_id];
//     Mat2d_list PEs = Q_PP[cell_id];
//     std::vector<double> Ls = Q_L[cell_id];
//     for(int j=0; j<adj_ids.size(); j++){
//       near_ids.insert(near_ids.end(),Q_PEI[adj_ids[j]].begin(),Q_PEI[adj_ids[j]].end());
//       PEs.insert(PEs.end(),Q_PP[adj_ids[j]].begin(),Q_PP[adj_ids[j]].end());
//       Ls.insert(Ls.end(),Q_L[adj_ids[j]].begin(),Q_L[adj_ids[j]].end());
//     }
//     Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());

//     Eigen::MatrixXd sigma_n=igl::slice(sigma,nI,1);
//     Eigen::MatrixXd mu_n=igl::slice(mu,nI,1);
//     Eigen::MatrixX2d N_n=igl::slice(N,nI,1);
//     Eigen::VectorXd L_n =igl::slice(L,nI);

//     Eigen::VectorXd Lsv=Eigen::Map<Eigen::VectorXd>(Ls.data(),Ls.size());
//     Lsv = Lsv.array()*L_n.array();
    
//     for(int ci=0; ci<Q_QI[cell_id].size(); ci++){
//       int qid = Q_QI[cell_id][ci];
//       std::complex<double> qi = Qi(qid);

//       if(near_ids.size()>0){
//         Eigen::RowVectorXd G;
//         Eigen::RowVectorXd F;
//         green_line_integral(PEs,N_n,Lsv,Q.row(qid),G,F);
//         W.row(qid) = G*sigma_n + F*mu_n;
//       }

//       Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
//       compute_Ik(qi-cni, num_expansion, Ik);

//       Eigen::MatrixXcd Ok=Eigen::MatrixXcd::Zero(l3_ids.size(), num_expansion+1);
//       for(int j=0; j<l3_ids.size(); j++){
//         if(Q_PEI[l3_ids[j]].size()==0){
//           continue;
//         }
//         compute_Ok(qi-Q_CNi(l3_ids[j]),(num_expansion+1),Ok,j);
//       }

//       //#pragma omp parallel for
//       for(int j=0; j<nj; j++){
//         Eigen::MatrixXcd& Mg = Mg_list[j];  Eigen::MatrixXcd& Mf = Mf_list[j];
//         Eigen::MatrixXcd& Lg = Lg_list[j];  Eigen::MatrixXcd& Lf = Lf_list[j];
//         for(int l=0; l<num_expansion; l++){
//           for(int k=0; k<l3_ids.size(); k++){
//             if(Q_PEI[l3_ids[k]].size()==0)
//               continue;
//             W(qid,j)=W(qid,j)+(Ok(k,l)*Mg(l3_ids[k],l)-Ok(k,l+1)*Mf(l3_ids[k],l)).real()/(2*igl::PI);
//           }
//           W(qid,j)=W(qid,j)+(Ik(l)*(Lg(cell_id,l)-Lf(cell_id,l))).real()/(2*igl::PI);
//         }
//       }
//     }
//   }
// }

// void infinite::eval_fmm_integral_uni(const Eigen::MatrixX2d &P,
//                        const Eigen::MatrixX2i &E,
//                        const Eigen::MatrixX2d &C,
//                        const Eigen::MatrixX2d &N,
//                        const Eigen::VectorXd &L,
//                        const Eigen::MatrixX2d &Q,
//                        const Eigen::MatrixXd &sigma,
//                        const Eigen::MatrixXd &mu,
//                        const int &num_expansion,
//                        const int &min_pnt_num,
//                        const int &max_depth,
//                        Eigen::MatrixXd &W)
// {
//   Eigen::MatrixX2d PQ(P.rows()+Q.rows(),P.cols());
//   PQ<<P,Q;

//   Eigen::RowVector2d minQ = PQ.colwise().minCoeff();
//   Eigen::RowVector2d maxQ = PQ.colwise().maxCoeff();

//   std::vector<std::vector<int > > Q_PI, Q_PEI, Q_QI;
//   std::vector<std::vector<double> > Q_L;
//   Eigen::MatrixXi Q_CH;
//   Eigen::VectorXi Q_PA, Q_LV;
//   Eigen::MatrixXd Q_CN;
//   Eigen::VectorXd Q_W;
//   std::vector<std::vector<int> > Q_adjs, Q_small_seps, Q_big_seps, Q_inters;
//   Mat2d_list_list Q_PP;
//   RowVec2d_list_list Q_LL;

//   quadtree(P,minQ,maxQ,max_depth,min_pnt_num,
//                 Q_PI,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);
//   compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_inters);

//   std::vector<int> leaf_cells;
//   for(int i=0; i<Q_CH.rows(); i++){
//     if(Q_CH(i,0)==-1){
//       leaf_cells.emplace_back(i);
//     }
//   }
//   std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
//   for (int i=0; i<Q_LV.size(); i++){
//     levels[Q_LV(i)].emplace_back(i);
//   }

//   Eigen::VectorXcd Ni = mat2d_to_compvec(N);
//   Eigen::VectorXcd Qi = mat2d_to_compvec(Q);
//   Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

//   int nj = sigma.cols();



//   MatXcd_list Lg_list(nj);
//   MatXcd_list Lf_list(nj);
//   MatXcd_list Mg_list(nj);
//   MatXcd_list Mf_list(nj);


//   // #pragma omp parallel for
//   for(int j=0; j<nj; j++){
//     Eigen::MatrixXcd Mg,Mf,Ug,Uf,Lg,Lf;

//     Mg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Ug=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Lg=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Mf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Uf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);
//     Lf=Eigen::MatrixXcd::Zero(Q_PEI.size(),num_expansion);

//     outgoing_from_source(sigma.col(j),mu.col(j),Ni,leaf_cells,Q_PEI,
//                           Ik_out_list_list,num_expansion,Mg,Mf);
//     outgoing_from_outgoing(levels,Q_PEI,Ik_child_list,Q_CH,num_expansion,Mg,Mf);

//     incoming_from_source(sigma.col(j),mu.col(j),Ni,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug,Uf);

//     incoming_from_outgoing(Q_PEI,Q_inters,Mg,Mf,num_expansion,Ok_inter_list_list,Lg,Lf);
//     Lg = Lg+Ug; Lf = Lf+Uf;
//     incoming_from_incoming(Q_PEI,levels,Q_CH,num_expansion,Ik_child_list,Lg,Lf);


//     Mg_list[j] = Mg;  Mf_list[j] = Mf;
//     Lg_list[j] = Lg;  Lf_list[j] = Lf;

//   }



//   eval_fmm_integral(sigma,mu,num_expansion,N,L,Q,Q_PEI,Q_QI,levels,Q_LL,
//           Q_PP,Q_CH,Q_LV,Q_CN,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,W);
// }



void infinite::eval_fmm_integral_uni(
                  const Eigen::MatrixXd &sigma,
                  const Eigen::MatrixXd &mu,
                  const int &num_expansion,
                  const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const std::vector<std::vector<int > > &Q_PI,
                  const std::vector<std::vector<int > > &Q_QI,
                  const std::vector<std::vector<int> > &levels,
                  const Eigen::MatrixXi &Q_CH,
                  const Eigen::VectorXi &Q_LV,
                  const Eigen::MatrixXd &Q_CN, 
                  const std::vector<std::vector<int> > &Q_adjs,
                  const std::vector<std::vector<int> > &Q_inters,
                  const std::vector<int> &leaf_cells,
                  Eigen::MatrixXd& W)
{
  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  Eigen::VectorXcd Qi = mat2d_to_compvec(Q);
  Eigen::VectorXcd Q_CNi = mat2d_to_compvec(Q_CN);

  int nj = sigma.cols();


  std::vector<VecXcd_list> Ik_out_list_list;
  std::vector<VecXcd_list> Ok_inter_list_list;
  MatXcd_list Ik_child_list;

  //need to add 
  //precompute_expansions_cell_dependent
  precompute_expansions_cell_dependent(P,E,N,L,num_expansion,Q_PI,Q_CH,
    Q_LV,Q_CN,Q_inters,Ik_out_list_list,Ok_inter_list_list,Ik_child_list);



  MatXcd_list Lg_list(nj);
  MatXcd_list Lf_list(nj);
  // MatXcd_list Mg_list(nj);
  // MatXcd_list Mf_list(nj);


  // #pragma omp parallel for
  for(int j=0; j<nj; j++){
    Eigen::MatrixXcd Mg,Mf,Lg,Lf;

    Mg=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Lg=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Mf=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);
    Lf=Eigen::MatrixXcd::Zero(Q_PI.size(),num_expansion);

    outgoing_from_source(sigma.col(j),mu.col(j),Ni,leaf_cells,Q_PI,
                          Ik_out_list_list,num_expansion,Mg,Mf);
    outgoing_from_outgoing(levels,Q_PI,Ik_child_list,Q_CH,num_expansion,Mg,Mf);

    // incoming_from_source(sigma.col(j),mu.col(j),Ni,Ok_inc_list_list,Q_big_seps,Q_PEI,num_expansion,Ug,Uf);

    incoming_from_outgoing(Q_PI,Q_inters,Mg,Mf,num_expansion,Ok_inter_list_list,Lg,Lf);
    // Lg = Lg+Ug; Lf = Lf+Uf;
    incoming_from_incoming_uni(Q_PI,levels,Q_CH,num_expansion,Ik_child_list,Lg,Lf);


    // Mg_list[j] = Mg;  Mf_list[j] = Mf;
    Lg_list[j] = Lg;  Lf_list[j] = Lf;

  }

  W = Eigen::MatrixXd::Zero(Q.rows(),nj);

  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int cell_id = leaf_cells[i];
    std::complex<double> cni = Q_CNi(cell_id);
    const std::vector<int> &adj_ids = Q_adjs[cell_id];

    std::vector<int> near_ids = Q_PI[cell_id];
    for(int j=0; j<adj_ids.size(); j++){
      near_ids.insert(near_ids.end(),Q_PI[adj_ids[j]].begin(),Q_PI[adj_ids[j]].end());
    }
    Eigen::VectorXi nI=Eigen::Map<Eigen::VectorXi>(near_ids.data(),near_ids.size());

    Eigen::MatrixXd sigma_n=igl::slice(sigma,nI,1);
    Eigen::MatrixXd mu_n=igl::slice(mu,nI,1);
    
    for(int ci=0; ci<Q_QI[cell_id].size(); ci++){
      int qid = Q_QI[cell_id][ci];
      std::complex<double> qi = Qi(qid);

      if(near_ids.size()>0){
        Eigen::RowVectorXd G;
        Eigen::RowVectorXd F;
        // green_line_integral(PEs,N_n,L_n,Q.row(qid),G,F);
        
        green_line_integral(P,E,N,L,near_ids,Q.row(qid),G,F);
        W.row(qid) = G*sigma_n + F*mu_n;
      }

      Eigen::VectorXcd Ik = Eigen::VectorXcd::Zero(num_expansion);
      compute_Ik(qi-cni, num_expansion, Ik);


      //#pragma omp parallel for
      for(int j=0; j<nj; j++){
        // Eigen::MatrixXcd& Mg = Mg_list[j];  Eigen::MatrixXcd& Mf = Mf_list[j];
        Eigen::MatrixXcd& Lg = Lg_list[j];  Eigen::MatrixXcd& Lf = Lf_list[j];
        for(int l=0; l<num_expansion; l++){
          // for(int k=0; k<l3_ids.size(); k++){
          //   if(Q_PEI[l3_ids[k]].size()==0)
          //     continue;
          //   W(qid,j)=W(qid,j)+(Ok(k,l)*Mg(l3_ids[k],l)-Ok(k,l+1)*Mf(l3_ids[k],l)).real()/(2*igl::PI);
          // }
          W(qid,j)=W(qid,j)+(Ik(l)*(Lg(cell_id,l)-Lf(cell_id,l))).real()/(2*igl::PI);
        }
      }
    }
  }

}


void infinite::eval_fmm_integral_uni(const Eigen::MatrixX2d &P,
                       const Eigen::MatrixX2i &E,
                       const Eigen::MatrixX2d &C,
                       const Eigen::MatrixX2d &N,
                       const Eigen::VectorXd &L,
                       const Eigen::MatrixX2d &Q,
                       const Eigen::MatrixXd &sigma,
                       const Eigen::MatrixXd &mu,
                       const int &num_expansion,
                       const int &set_depth,
                       Eigen::MatrixXd &W)
{
  Eigen::MatrixX2d PQ(P.rows()+Q.rows(),P.cols());
  PQ<<P,Q;

  Eigen::RowVector2d minQ = PQ.colwise().minCoeff();
  Eigen::RowVector2d maxQ = PQ.colwise().maxCoeff();

  std::vector<std::vector<int > > Q_PI, Q_QI;
  Eigen::MatrixXi Q_CH;
  Eigen::VectorXi Q_PA, Q_LV;
  Eigen::MatrixXd Q_CN;
  Eigen::VectorXd Q_W;
  std::vector<std::vector<int> > Q_adjs, Q_inters;

  quadtree_uniform(C,Q,minQ,maxQ,set_depth,Q_PI,Q_QI,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);

  compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_inters);



  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  eval_fmm_integral_uni(sigma,mu,num_expansion,P,E,N,L,Q,Q_PI,Q_QI,levels,
          Q_CH,Q_LV,Q_CN,Q_adjs,Q_inters,leaf_cells,W);
}
