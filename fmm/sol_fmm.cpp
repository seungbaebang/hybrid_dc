#include "sol_fmm.h"
#include <igl/writeDMAT.h>

void infinite::local_resolve_curve(const int &num_expansion,
                  const Eigen::VectorXi &qI,
                  const Eigen::SparseMatrix<double> &VU,
                  const std::vector<std::vector<int > > &Q_PEI,
                  const std::vector<std::vector<int> > &levels,
                  const Eigen::MatrixXi &Q_CH,
                  const std::vector<std::vector<int> > &Q_adjs,
                  const std::vector<std::vector<int> > &Q_small_seps,
                  const std::vector<std::vector<int> > &Q_inters,
                  const std::vector<std::vector<int> > &Q_big_seps,
                  const std::vector<int> &leaf_cells,
                  const Eigen::VectorXi &newI,
                  const Eigen::VectorXi &new_pids,
                  const int &nb,
                  const int &ns,
                  const std::vector<VecXcd_list> &Ik_out_list_list,
                  const std::vector<VecXcd_list> &Ok_inter_list_list,
                  const MatXcd_list &Ik_child_list,
                  const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  const std::vector<RowVecXd_list> &G_list,
                  const std::vector<RowVecXd_list> &F_list,
                  const std::vector<VecXi_list> &nI_list,
                  const std::vector<VecXcd_list> &Ok_l3_list,
                  const VecXcd_list &Ik_inc_list,
                  Eigen::VectorXi &sol_qids,
                  std::vector<bool> &sol_pids_bool,
                  std::vector<bool> &con_pids_bool)
{
  int ng = qI.size()/nb;


  std::vector<int> sol_list;
  Eigen::VectorXi qids = Eigen::VectorXi::LinSpaced(ng*nb,0,ng*nb-1);

  for(int i=0; i<newI.size();i++){
    sol_list.emplace_back(newI(i));
  }

  for(int i=0; i<newI.size(); i++){

    Eigen::VectorXd sol_sig = Eigen::VectorXd::Zero(nb*ns);
    std::vector<bool> pids_bool(nb*ns,false);

    for(int j=0; j<ns; j++){
      sol_sig(ns*newI(i)+j)=-1;
      pids_bool[ns*newI(i)+j]=true;
    }

    
    Eigen::VectorXd sol_eff;
    update_forward_fmm_g_vec(num_expansion,qI,qids,sol_sig,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,
      Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,G_list,nI_list,Ok_l3_list,Ik_inc_list,pids_bool,sol_eff);
    sol_eff = sol_eff.array()-sol_eff.minCoeff();

    double threshold_eff = 10*sol_eff.segment(ng*newI(i),ng).maxCoeff();

    for(int k=0; k<nb; k++){

      if(std::find(sol_list.begin(),sol_list.end(),k)!=sol_list.end())
        continue;

      for(int j=0; j<ng; j++){
        if(sol_eff(ng*k+j)<threshold_eff){
          sol_list.emplace_back(k);
          break;
        }
      }
    }
  }
  std::vector<int> u_sol_list;
  igl::unique(sol_list,u_sol_list);
  sol_list = u_sol_list;

  // solI = Eigen::Map<Eigen::VectorXi>(sol_list.data(),sol_list.size());

  sol_qids.resize(ng*sol_list.size());

  sol_pids_bool.resize(nb*ns,false);
  con_pids_bool.resize(nb*ns,true);


  for(int i=0; i<sol_list.size(); i++){
    for(int j=0; j<ns; j++){
      sol_pids_bool[ns*sol_list[i]+j] = true;
      con_pids_bool[ns*sol_list[i]+j] = false;
    }
    for(int j=0; j<ng; j++){
      sol_qids(ng*i+j) = ng*sol_list[i]+j;
    }
  }
  for(int i=0; i<new_pids.size(); i++){
    sol_pids_bool[new_pids(i)]=true;
    con_pids_bool[new_pids(i)]=false;
  }
}


void infinite::local_resolve_curve(const int &num_expansion,
                  const Eigen::VectorXi &qI,
                  const Eigen::SparseMatrix<double> &VU,
                  const std::vector<std::vector<int > > &Q_PEI,
                  const std::vector<std::vector<int> > &levels,
                  const Eigen::MatrixXi &Q_CH,
                  const std::vector<std::vector<int> > &Q_adjs,
                  const std::vector<std::vector<int> > &Q_small_seps,
                  const std::vector<std::vector<int> > &Q_inters,
                  const std::vector<std::vector<int> > &Q_big_seps,
                  const std::vector<int> &leaf_cells,
                  const Eigen::VectorXi &newI,
                  const Eigen::VectorXi &new_pids,
                  const Eigen::VectorXi &new_qids,
                  const int &nb,
                  const int &ns,
                  const std::vector<VecXcd_list> &Ik_out_list_list,
                  const std::vector<VecXcd_list> &Ok_inter_list_list,
                  const MatXcd_list &Ik_child_list,
                  const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  const std::vector<RowVecXd_list> &G_list,
                  const std::vector<RowVecXd_list> &F_list,
                  const std::vector<VecXi_list> &nI_list,
                  const std::vector<VecXcd_list> &Ok_l3_list,
                  const VecXcd_list &Ik_inc_list,
                  Eigen::VectorXi &solI,
                  Eigen::VectorXi &sol_qids,
                  std::vector<bool> &sol_pids_bool,
                  std::vector<bool> &con_pids_bool)
{
  int ng = qI.size()/nb;


  std::vector<int> sol_list;
  Eigen::VectorXi qids = Eigen::VectorXi::LinSpaced(ng*nb,0,ng*nb-1);

  for(int i=0; i<newI.size();i++){
    sol_list.emplace_back(newI(i));
  }

  for(int i=0; i<newI.size(); i++){

    Eigen::VectorXd sol_sig = Eigen::VectorXd::Zero(nb*ns);
    std::vector<bool> pids_bool(nb*ns,false);

    for(int j=0; j<ns; j++){
      sol_sig(ns*newI(i)+j)=-1;
      pids_bool[ns*newI(i)+j]=true;
    }

    
    Eigen::VectorXd sol_eff;
    update_forward_fmm_g_vec(num_expansion,qI,qids,sol_sig,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,
      Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,G_list,nI_list,Ok_l3_list,Ik_inc_list,pids_bool,sol_eff);
    sol_eff = sol_eff.array()-sol_eff.minCoeff();

    double threshold_eff = 10*sol_eff.segment(ng*newI(i),ng).maxCoeff();

    for(int k=0; k<nb; k++){

      if(std::find(sol_list.begin(),sol_list.end(),k)!=sol_list.end())
        continue;

      for(int j=0; j<ng; j++){
        if(sol_eff(ng*k+j)<threshold_eff){
          sol_list.emplace_back(k);
          break;
        }
      }
    }
  }
  std::vector<int> u_sol_list;
  igl::unique(sol_list,u_sol_list);
  sol_list = u_sol_list;

  solI = Eigen::Map<Eigen::VectorXi>(sol_list.data(),sol_list.size());

  sol_qids.resize(ng*sol_list.size());

  sol_pids_bool.resize(nb*ns,false);
  con_pids_bool.resize(nb*ns,true);


  for(int i=0; i<sol_list.size(); i++){
    for(int j=0; j<ns; j++){
      sol_pids_bool[ns*sol_list[i]+j] = true;
      con_pids_bool[ns*sol_list[i]+j] = false;
    }
    for(int j=0; j<ng; j++){
      sol_qids(ng*i+j) = ng*sol_list[i]+j;
    }
  }
  for(int i=0; i<new_pids.size(); i++){
    sol_pids_bool[new_pids(i)]=true;
    con_pids_bool[new_pids(i)]=false;
  }
}




bool infinite::fmm_gmres(const Eigen::VectorXd &rhs, 
               const int &restart,
               const double &tol,
               const int &maxIters,
               const int &num_expansion,
               const std::vector<int>& leaf_cells,
               const std::vector<std::vector<int > > &Q_PEI,
               const std::vector<std::vector<int> > &levels,
               const Eigen::MatrixXi &Q_CH,
               const std::vector<std::vector<int> > &Q_adjs,
               const std::vector<std::vector<int> > &Q_small_seps,
               const std::vector<std::vector<int> > &Q_big_seps,
               const std::vector<std::vector<int> > &Q_inters,
               const std::vector<RowVecXd_list> &G_list,
               const std::vector<VecXi_list> &nI_list,
               const std::vector<VecXcd_list> &Ok_l3_list,
               const VecXcd_list &Ik_inc_list,
               const std::vector<VecXcd_list> &Ik_out_list,
               const std::vector<VecXcd_list> &Ok_inter_list,
               const MatXcd_list &Ik_child_list,
               const std::vector<std::vector<VecXcd_list> > &Ok_inc_list,
               const Eigen::VectorXi &qI,
               const double &r0Norm,
               Eigen::VectorXd &x)
{

  double tol_error = tol;
  int iters = 0;
  int m = rhs.size();

  Eigen::VectorXd b;

  forward_fmm_g_vec(num_expansion,qI,x,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
    nI_list,Ok_l3_list,Ik_inc_list,b);

  Eigen::VectorXd p0 = rhs - b;
  Eigen::VectorXd r0 = p0;

  if(r0Norm==0)
  {
    tol_error=0;
    return true;
  }

  Eigen::MatrixXd H   = Eigen::MatrixXd::Zero(m, restart + 1);
  Eigen::VectorXd w   = Eigen::VectorXd::Zero(restart + 1);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(restart + 1);

  std::vector < Eigen::JacobiRotation < double > > G(restart);

  Eigen::VectorXd t(m), v(m), workspace(m), x_new(m);

  Eigen::Ref<Eigen::VectorXd> H0_tail = H.col(0).tail(m-1);

  double beta;
  r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
  w(0) = beta;
  for(int k=1; k<=restart; ++k)
  { 
    std::cout<<"iter: "<<iters;
    ++iters;
    v = Eigen::VectorXd::Unit(m,k-1);

    for(int i=k-1; i>=0; --i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }

    forward_fmm_g_vec(num_expansion,qI,v,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
      Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
      nI_list,Ok_l3_list,Ik_inc_list,b);

    t.noalias() = b;
    v=t;
    for(int i=0; i<k; ++i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }
    if(v.tail(m-k).norm() != 0.0)
    {
      if(k<=restart)
      {
        Eigen::Ref<Eigen::VectorXd> Hk_tail = H.col(k).tail(m-k-1);
        v.tail(m-k).makeHouseholder(Hk_tail,tau.coeffRef(k),beta);
        v.tail(m-k).applyHouseholderOnTheLeft(Hk_tail,tau.coeffRef(k),workspace.data());
      }
    }

    if(k>1)
    {
      for(int i=0; i<k-1; ++i)
      {
        v.applyOnTheLeft(i,i+1,G[i].adjoint());
      }
    }
    if(k<m && v(k) != 0)
    {
      G[k-1].makeGivens(v(k-1),v(k));
      v.applyOnTheLeft(k-1,k,G[k-1].adjoint());
      w.applyOnTheLeft(k-1,k,G[k-1].adjoint());
    }

    H.col(k-1).head(k) = v.head(k);

    tol_error = std::abs(w(k)) / r0Norm;
    // tol_error = (rhs - b).norm() / r0Norm;
    bool stop = (k==m || tol_error < tol || iters == maxIters);
    std::cout<<", err: "<<tol_error<<std::endl;


    if(stop || k== restart)
    {
      Eigen::Ref<Eigen::VectorXd> y = w.head(k);
      H.topLeftCorner(k,k).template triangularView <Eigen::Upper>().solveInPlace(y);

      x_new.setZero();
      for(int i=k-1; i>=0; --i)
      {
        x_new(i) += y(i);
        x_new.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
      }
      x += x_new;
      if(stop)
      {
        // forward_fmm_g_vec(num_expansion,qI,x,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
        //   Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
        //   nI_list,Ok_l3_list,Ik_inc_list,b);
        return true;
      }
      else
      {
        k=0;
        forward_fmm_g_vec(num_expansion,qI,x,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
          Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
          nI_list,Ok_l3_list,Ik_inc_list,b);

        p0.noalias() = rhs - b;
        r0 = p0;

        H.setZero();
        w.setZero();
        tau.setZero();

        r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
        w(0) = beta;
      }
    }
  }
  return false;
}

bool infinite::fmm_gmres_hybrid(const Eigen::VectorXd &rhs, 
               const int &restart,
               const double &tol,
               const int &maxIters,
               const int &num_expansion,
               const std::vector<int>& leaf_cells,
               const std::vector<std::vector<int > > &Q_PEI,
               const std::vector<std::vector<int> > &levels,
               const Eigen::MatrixXi &Q_CH,
               const std::vector<std::vector<int> > &Q_adjs,
               const std::vector<std::vector<int> > &Q_small_seps,
               const std::vector<std::vector<int> > &Q_big_seps,
               const std::vector<std::vector<int> > &Q_inters,
               const std::vector<RowVecXd_list> &G_list,
               const std::vector<VecXi_list> &nI_list,
               const std::vector<VecXcd_list> &Ok_l3_list,
               const VecXcd_list &Ik_inc_list,
               const std::vector<VecXcd_list> &Ik_out_list,
               const std::vector<VecXcd_list> &Ok_inter_list,
               const MatXcd_list &Ik_child_list,
               const std::vector<std::vector<VecXcd_list> > &Ok_inc_list,
               const Eigen::VectorXi &qI,
               const Eigen::SparseMatrix<double> &VU,
               const double &r0Norm,
               Eigen::VectorXd &x,
               std::vector<Eigen::VectorXd> &x_list,
               std::vector<double> &err_list)
{

  double tol_error = tol;
  int iters = 0;
  int m = rhs.size();

  Eigen::VectorXd b;
  Eigen::VectorXd x_s = VU*x;

  forward_fmm_g_vec(num_expansion,qI,x_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
    nI_list,Ok_l3_list,Ik_inc_list,b);


  Eigen::VectorXd p0 = rhs - b;
  Eigen::VectorXd r0 = p0;

  if(r0Norm==0)
  {
    tol_error=0;
    return true;
  }

  Eigen::MatrixXd H   = Eigen::MatrixXd::Zero(m, restart + 1);
  Eigen::VectorXd w   = Eigen::VectorXd::Zero(restart + 1);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(restart + 1);

  std::vector < Eigen::JacobiRotation < double > > G(restart);

  Eigen::VectorXd t(m), v(m), workspace(m), x_new(m);

  Eigen::Ref<Eigen::VectorXd> H0_tail = H.col(0).tail(m-1);

  double beta;
  r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
  w(0) = beta;
  for(int k=1; k<=restart; ++k)
  { 
    std::cout<<"iter: "<<iters;
    ++iters;
    v = Eigen::VectorXd::Unit(m,k-1);

    for(int i=k-1; i>=0; --i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }
    Eigen::VectorXd v_s = VU*v;

    forward_fmm_g_vec(num_expansion,qI,v_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
      Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
      nI_list,Ok_l3_list,Ik_inc_list,b);

    t.noalias() = b;
    v=t;
    for(int i=0; i<k; ++i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }
    if(v.tail(m-k).norm() != 0.0)
    {
      if(k<=restart)
      {
        Eigen::Ref<Eigen::VectorXd> Hk_tail = H.col(k).tail(m-k-1);
        v.tail(m-k).makeHouseholder(Hk_tail,tau.coeffRef(k),beta);
        v.tail(m-k).applyHouseholderOnTheLeft(Hk_tail,tau.coeffRef(k),workspace.data());
      }
    }

    if(k>1)
    {
      for(int i=0; i<k-1; ++i)
      {
        v.applyOnTheLeft(i,i+1,G[i].adjoint());
      }
    }
    if(k<m && v(k) != 0)
    {
      G[k-1].makeGivens(v(k-1),v(k));
      v.applyOnTheLeft(k-1,k,G[k-1].adjoint());
      w.applyOnTheLeft(k-1,k,G[k-1].adjoint());
    }

    H.col(k-1).head(k) = v.head(k);

    tol_error = std::abs(w(k)) / r0Norm;
    // tol_error = (rhs - b).norm() / r0Norm;
    bool stop = (k==m || tol_error < tol || iters == maxIters);
    std::cout<<", err: "<<tol_error<<std::endl;
    {
      Eigen::VectorXd w1 = w;
      Eigen::Ref<Eigen::VectorXd> y1 = w1.head(k);
      Eigen::MatrixXd  H1=H;
      H1.topLeftCorner(k,k).template triangularView <Eigen::Upper>().solveInPlace(y1);

      Eigen::VectorXd tau1 = tau;
      Eigen::VectorXd workspace1 = workspace;

      Eigen::VectorXd x_new1(m);
      x_new1.setZero();
      for(int i=k-1; i>=0; --i)
      {
        x_new1(i) += y1(i);
        x_new1.tail(m-i).applyHouseholderOnTheLeft(H1.col(i).tail(m-i-1),tau1.coeffRef(i),workspace1.data());
      }
      Eigen::VectorXd x1 = x;
      x1 += x_new1;
      x_list.emplace_back(x1);
    }
    err_list.emplace_back(tol_error);

    if(stop || k== restart)
    {
      Eigen::Ref<Eigen::VectorXd> y = w.head(k);
      H.topLeftCorner(k,k).template triangularView <Eigen::Upper>().solveInPlace(y);

      x_new.setZero();
      for(int i=k-1; i>=0; --i)
      {
        x_new(i) += y(i);
        x_new.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
      }
      x += x_new;
      if(stop)
      {
        return true;
      }
      else
      {
        k=0;

        Eigen::VectorXd x_s = VU*x;
        forward_fmm_g_vec(num_expansion,qI,x_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
          Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
          nI_list,Ok_l3_list,Ik_inc_list,b);

        p0.noalias() = rhs - b;
        r0 = p0;

        H.setZero();
        w.setZero();
        tau.setZero();

        r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
        w(0) = beta;
      }
    }
  }
  return false;
}


bool infinite::fmm_gmres_hybrid(const Eigen::VectorXd &rhs, 
               const int &restart,
               const double &tol,
               const int &maxIters,
               const int &num_expansion,
               const std::vector<int>& leaf_cells,
               const std::vector<std::vector<int > > &Q_PEI,
               const std::vector<std::vector<int> > &levels,
               const Eigen::MatrixXi &Q_CH,
               const std::vector<std::vector<int> > &Q_adjs,
               const std::vector<std::vector<int> > &Q_small_seps,
               const std::vector<std::vector<int> > &Q_big_seps,
               const std::vector<std::vector<int> > &Q_inters,
               const std::vector<RowVecXd_list> &G_list,
               const std::vector<VecXi_list> &nI_list,
               const std::vector<VecXcd_list> &Ok_l3_list,
               const VecXcd_list &Ik_inc_list,
               const std::vector<VecXcd_list> &Ik_out_list,
               const std::vector<VecXcd_list> &Ok_inter_list,
               const MatXcd_list &Ik_child_list,
               const std::vector<std::vector<VecXcd_list> > &Ok_inc_list,
               const Eigen::VectorXi &qI,
               const Eigen::SparseMatrix<double> &VU,
               const double &r0Norm,
               Eigen::VectorXd &x)
{

  double tol_error = tol;
  int iters = 0;
  int m = rhs.size();

  Eigen::VectorXd b;
  Eigen::VectorXd x_s = VU*x;

  forward_fmm_g_vec(num_expansion,qI,x_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
    nI_list,Ok_l3_list,Ik_inc_list,b);


  Eigen::VectorXd p0 = rhs - b;
  Eigen::VectorXd r0 = p0;

  if(r0Norm==0)
  {
    tol_error=0;
    return true;
  }

  Eigen::MatrixXd H   = Eigen::MatrixXd::Zero(m, restart + 1);
  Eigen::VectorXd w   = Eigen::VectorXd::Zero(restart + 1);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(restart + 1);

  std::vector < Eigen::JacobiRotation < double > > G(restart);

  Eigen::VectorXd t(m), v(m), workspace(m), x_new(m);

  Eigen::Ref<Eigen::VectorXd> H0_tail = H.col(0).tail(m-1);

  double beta;
  r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
  w(0) = beta;
  for(int k=1; k<=restart; ++k)
  { 
    std::cout<<"iter: "<<iters;
    ++iters;
    v = Eigen::VectorXd::Unit(m,k-1);

    for(int i=k-1; i>=0; --i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }
    Eigen::VectorXd v_s = VU*v;

    forward_fmm_g_vec(num_expansion,qI,v_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
      Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
      nI_list,Ok_l3_list,Ik_inc_list,b);

    t.noalias() = b;
    v=t;
    for(int i=0; i<k; ++i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }
    if(v.tail(m-k).norm() != 0.0)
    {
      if(k<=restart)
      {
        Eigen::Ref<Eigen::VectorXd> Hk_tail = H.col(k).tail(m-k-1);
        v.tail(m-k).makeHouseholder(Hk_tail,tau.coeffRef(k),beta);
        v.tail(m-k).applyHouseholderOnTheLeft(Hk_tail,tau.coeffRef(k),workspace.data());
      }
    }

    if(k>1)
    {
      for(int i=0; i<k-1; ++i)
      {
        v.applyOnTheLeft(i,i+1,G[i].adjoint());
      }
    }
    if(k<m && v(k) != 0)
    {
      G[k-1].makeGivens(v(k-1),v(k));
      v.applyOnTheLeft(k-1,k,G[k-1].adjoint());
      w.applyOnTheLeft(k-1,k,G[k-1].adjoint());
    }

    H.col(k-1).head(k) = v.head(k);

    tol_error = std::abs(w(k)) / r0Norm;
    // tol_error = (rhs - b).norm() / r0Norm;
    bool stop = (k==m || tol_error < tol || iters == maxIters);
    std::cout<<", err: "<<tol_error<<std::endl;


    if(stop || k== restart)
    {
      Eigen::Ref<Eigen::VectorXd> y = w.head(k);
      H.topLeftCorner(k,k).template triangularView <Eigen::Upper>().solveInPlace(y);

      x_new.setZero();
      for(int i=k-1; i>=0; --i)
      {
        x_new(i) += y(i);
        x_new.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
      }
      x += x_new;
      if(stop)
      {
        return true;
      }
      else
      {
        k=0;

        Eigen::VectorXd x_s = VU*x;
        forward_fmm_g_vec(num_expansion,qI,x_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
          Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,G_list,
          nI_list,Ok_l3_list,Ik_inc_list,b);

        p0.noalias() = rhs - b;
        r0 = p0;

        H.setZero();
        w.setZero();
        tau.setZero();

        r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
        w(0) = beta;
      }
    }
  }
  return false;
}


bool infinite::update_fmm_gmres_hybrid(
               const Eigen::VectorXd &rhs, 
               const int &restart,
               const double &tol,
               const int &maxIters,
               const int &num_expansion,
               const std::vector<int>& leaf_cells,
               const std::vector<std::vector<int > > &Q_PEI,
               const std::vector<std::vector<int> > &levels,
               const Eigen::MatrixXi &Q_CH,
               const std::vector<std::vector<int> > &Q_adjs,
               const std::vector<std::vector<int> > &Q_small_seps,
               const std::vector<std::vector<int> > &Q_big_seps,
               const std::vector<std::vector<int> > &Q_inters,
               const std::vector<RowVecXd_list> &G_list,
               const std::vector<VecXi_list> &nI_list,
               const std::vector<VecXcd_list> &Ok_l3_list,
               const VecXcd_list &Ik_inc_list,
               const std::vector<VecXcd_list> &Ik_out_list,
               const std::vector<VecXcd_list> &Ok_inter_list,
               const MatXcd_list &Ik_child_list,
               const std::vector<std::vector<VecXcd_list> > &Ok_inc_list,
               const Eigen::VectorXi &qI,
               const Eigen::SparseMatrix<double> &VU,
               const double &r0Norm,
               const std::vector<bool> &update_pids,
               const Eigen::VectorXi &sol_qids,
               Eigen::VectorXd &x)
{

  double tol_error = tol;
  int iters = 0;
  int m = rhs.size();

  Eigen::VectorXd b;
  Eigen::VectorXd x_s = VU*x;
  update_forward_fmm_g_vec(num_expansion,qI,sol_qids,x_s,leaf_cells,levels,Q_PEI,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,
    G_list,nI_list,Ok_l3_list,Ik_inc_list,update_pids,b);

  Eigen::VectorXd p0 = rhs - b;
  Eigen::VectorXd r0 = p0;

  if(r0Norm==0)
  {
    tol_error=0;
    return true;
  }

  Eigen::MatrixXd H   = Eigen::MatrixXd::Zero(m, restart + 1);
  Eigen::VectorXd w   = Eigen::VectorXd::Zero(restart + 1);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(restart + 1);

  std::vector < Eigen::JacobiRotation < double > > G(restart);

  Eigen::VectorXd t(m), v(m), workspace(m), x_new(m);

  Eigen::Ref<Eigen::VectorXd> H0_tail = H.col(0).tail(m-1);

  double beta;
  r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
  w(0) = beta;
  for(int k=1; k<=restart; ++k)
  { 
    std::cout<<"iter: "<<iters;
    ++iters;
    v = Eigen::VectorXd::Unit(m,k-1);

    for(int i=k-1; i>=0; --i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }
    Eigen::VectorXd vn=Eigen::VectorXd::Zero(x.size());
    igl::slice_into(v,sol_qids,vn);
    Eigen::VectorXd v_s = VU*vn;
    update_forward_fmm_g_vec(num_expansion,qI,sol_qids,v_s,leaf_cells,levels,Q_PEI,
      Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,
      G_list,nI_list,Ok_l3_list,Ik_inc_list,update_pids,b);


    t.noalias() = b;
    v=t;
    for(int i=0; i<k; ++i){
      v.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
    }
    if(v.tail(m-k).norm() != 0.0)
    {
      if(k<=restart)
      {
        Eigen::Ref<Eigen::VectorXd> Hk_tail = H.col(k).tail(m-k-1);
        v.tail(m-k).makeHouseholder(Hk_tail,tau.coeffRef(k),beta);
        v.tail(m-k).applyHouseholderOnTheLeft(Hk_tail,tau.coeffRef(k),workspace.data());
      }
    }

    if(k>1)
    {
      for(int i=0; i<k-1; ++i)
      {
        v.applyOnTheLeft(i,i+1,G[i].adjoint());
      }
    }
    if(k<m && v(k) != 0)
    {
      G[k-1].makeGivens(v(k-1),v(k));
      v.applyOnTheLeft(k-1,k,G[k-1].adjoint());
      w.applyOnTheLeft(k-1,k,G[k-1].adjoint());
    }

    H.col(k-1).head(k) = v.head(k);

    tol_error = std::abs(w(k)) / r0Norm;
    // tol_error = (rhs - b).norm() / r0Norm;
    bool stop = (k==m || tol_error < tol || iters == maxIters);
    std::cout<<", err: "<<tol_error<<std::endl;


    if(stop || k== restart)
    {
      Eigen::Ref<Eigen::VectorXd> y = w.head(k);
      H.topLeftCorner(k,k).template triangularView <Eigen::Upper>().solveInPlace(y);

      x_new.setZero();
      for(int i=k-1; i>=0; --i)
      {
        x_new(i) += y(i);
        x_new.tail(m-i).applyHouseholderOnTheLeft(H.col(i).tail(m-i-1),tau.coeffRef(i),workspace.data());
      }
      // x += x_new;
      Eigen::VectorXd x_new1=Eigen::VectorXd::Zero(x.size());
      igl::slice_into(x_new,sol_qids,x_new1);
      x += x_new1;
      if(stop)
      {
        // update_forward_fmm_g_vec(num_expansion,qI,sol_qids,x,VU,leaf_cells,levels,Q_PEI,
        //   Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,
        //   G_list,nI_list,Ok_l3_list,Ik_inc_list,update_pids,b);
        return true;
      }
      else
      {
        k=0;
        Eigen::VectorXd x_s = VU*x;
        update_forward_fmm_g_vec(num_expansion,qI,sol_qids,x_s,leaf_cells,levels,Q_PEI,
          Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_CH,Ik_out_list,Ok_inter_list,Ik_child_list,Ok_inc_list,
          G_list,nI_list,Ok_l3_list,Ik_inc_list,update_pids,b);

        p0.noalias() = rhs - b;
        r0 = p0;

        H.setZero();
        w.setZero();
        tau.setZero();

        r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
        w(0) = beta;
      }
    }
  }
  return false;
}


void infinite::solve_fmm_gmres_hybrid(const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const int &num_expansion,
                  const Eigen::VectorXi &qI,
                  const Eigen::SparseMatrix<double> &VU,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const VecXi_list &sing_id_list,
                  const VecXd_list &sing_t_list,
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
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma,
                  std::vector<Eigen::MatrixXd> &sigma_list,
                  std::vector<Eigen::VectorXd> &err_list)
{
  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  pre_fmm(N,L,Q,qI,sing_id_list,sing_t_list,num_expansion,Q_PEI,Q_QI,Q_LL,
    Q_PP,Q_CH,Q_LV,Q_CN,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,
    Ik_child_list,Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list);

  Eigen::MatrixXd Wf;
  Eigen::MatrixXd mu_s = VU*mu;
  forward_fmm_f(Ni,num_expansion,qI,mu_s,leaf_cells,levels,Q_PEI,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,Wf);



  Eigen::MatrixXd bn = b-Wf;
  
  int nj = bn.cols();

  if(sigma.rows()<1){
    sigma = Eigen::MatrixXd::Ones(qI.size(),nj);
  }
  std::vector<std::vector<Eigen::VectorXd> > x_list_list(nj);
  std::vector<std::vector<double> > err_list_list(nj);

  int max_iter=0;
  for(int j=0; j<nj; j++){
    Eigen::VectorXd x = sigma.col(j);

    std::vector<Eigen::VectorXd> x_list;
    std::vector<double> err_list;

    fmm_gmres_hybrid(bn.col(j),restart,tol,maxIters,num_expansion,leaf_cells,Q_PEI,levels,Q_CH,
              Q_adjs,Q_small_seps,Q_big_seps,Q_inters,G_list,nI_list,Ok_l3_list,Ik_inc_list,Ik_out_list_list,Ok_inter_list_list,
              Ik_child_list,Ok_inc_list_list,qI,VU,b_norm(j),x,x_list,err_list);
    max_iter = std::max(max_iter, (int)x_list.size());


    x_list_list[j]=x_list;
    err_list_list[j]=err_list;

    sigma.col(j) = x;
  }
  sigma_list.resize(max_iter);
  err_list.resize(max_iter);

  for(int i=0; i<max_iter; i++){
    Eigen::MatrixXd X(sigma.rows(),nj);
    Eigen::VectorXd ER = Eigen::VectorXd(nj);
    for(int j=0; j<nj; j++){
      int id;
      if(i<x_list_list[j].size()){
        id=i; 
      }else{
        id=x_list_list[j].size()-1;
      }
      X.col(j)=x_list_list[j][id];
      ER(j) = err_list_list[j][id];
    }
    sigma_list[i]=X;
    err_list[i]=ER;
  }

}



void infinite::solve_fmm_gmres_hybrid(const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const int &num_expansion,
                  const Eigen::VectorXi &qI,
                  const Eigen::SparseMatrix<double> &VU,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const VecXi_list &sing_id_list,
                  const VecXd_list &sing_t_list,
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
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma)
{
  Eigen::VectorXcd Ni = mat2d_to_compvec(N);
  pre_fmm(N,L,Q,qI,sing_id_list,sing_t_list,num_expansion,Q_PEI,Q_QI,Q_LL,
    Q_PP,Q_CH,Q_LV,Q_CN,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,
    Ik_child_list,Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list);

  Eigen::MatrixXd Wf;
  Eigen::MatrixXd mu_s = VU*mu;
  forward_fmm_f(Ni,num_expansion,qI,mu_s,leaf_cells,levels,Q_PEI,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,Wf);

  Eigen::MatrixXd bn = b-Wf;
  
  int nj = bn.cols();

  if(sigma.rows()<1){
    sigma = Eigen::MatrixXd::Ones(qI.size(),nj);
  }
  for(int j=0; j<nj; j++){
    Eigen::VectorXd x = sigma.col(j);

    fmm_gmres_hybrid(bn.col(j),restart,tol,maxIters,num_expansion,leaf_cells,Q_PEI,levels,Q_CH,
              Q_adjs,Q_small_seps,Q_big_seps,Q_inters,G_list,nI_list,Ok_l3_list,Ik_inc_list,Ik_out_list_list,Ok_inter_list_list,
              Ik_child_list,Ok_inc_list_list,qI,VU,b_norm(j),x);

    sigma.col(j) = x;
  }
}


void infinite::resolve_fmm_gmres_hybrid(const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const int &num_expansion,
                  const Eigen::VectorXi &qI,
                  const Eigen::SparseMatrix<double> &VU,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const VecXi_list &sing_id_list,
                  const VecXd_list &sing_t_list,
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
                  const std::vector<std::vector<int> > &Q_inters_a,
                  const std::vector<int> &leaf_cells,
                  const Eigen::VectorXi &sub_cells,
                  const Eigen::VectorXi &new_cells,
                  const Eigen::VectorXi &newI,
                  const Eigen::VectorXi &new_pids,
                  const Eigen::VectorXi &sub_qids,
                  const Eigen::VectorXi &new_qids,
                  const Eigen::VectorXi &inI,
                  const std::vector<int> &update_leafs,
                  const std::vector<int> &update_big_seps,
                  const int &nb,
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma,
                  Eigen::VectorXi &solI)
{

  update_pre_fmm_hybrid(N,L,Q,sing_id_list,sing_t_list,num_expansion,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    sub_cells,new_cells,leaf_cells,Q_adjs,Q_small_seps,Q_big_seps,Q_inters_a,update_leafs,
    update_big_seps,new_pids,sub_qids,new_qids,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list);

  Eigen::VectorXcd Ni = mat2d_to_compvec(N);


  int ns = L.size()/nb;
  Eigen::VectorXi sol_qids;
  std::vector<bool> sol_pids_bool;
  std::vector<bool> con_pids_bool;
  local_resolve_curve(num_expansion,qI,VU,Q_PEI,levels,Q_CH,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,
    newI,new_pids,new_qids,nb,ns,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,G_list,
    F_list,nI_list,Ok_l3_list,Ik_inc_list,solI,sol_qids,sol_pids_bool,con_pids_bool);


  Eigen::MatrixXd Wf_i, Wf_b, Wg_b;

  Eigen::MatrixXd mu_s = VU*mu;
  Eigen::MatrixXd sigma_s = VU*sigma;

  update_forward_fmm_f(Ni,num_expansion,qI,sol_qids,mu_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,F_list,
    nI_list,Ok_l3_list,Ik_inc_list,sol_pids_bool,Wf_i);

  update_forward_fmm_f(Ni,num_expansion,qI,sol_qids,mu_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,F_list,
    nI_list,Ok_l3_list,Ik_inc_list,con_pids_bool,Wf_b);

  update_forward_fmm_g(num_expansion,qI,sol_qids,sigma_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,G_list,
    nI_list,Ok_l3_list,Ik_inc_list,con_pids_bool,Wg_b);

  Eigen::MatrixXd b_s = igl::slice(b,sol_qids,1);
  Eigen::MatrixXd bn = b_s - Wf_b - Wf_i - Wg_b;
  Eigen::VectorXd b_norm2 = b_s.colwise().norm();

  int nj = bn.cols();

  if(sigma.rows()<1){
    sigma = Eigen::MatrixXd::Ones(qI.size(),nj);
  }
  for(int j=0; j<nj; j++){
    Eigen::VectorXd x = sigma.col(j);
    
    update_fmm_gmres_hybrid(bn.col(j),restart,tol,maxIters,num_expansion,leaf_cells,Q_PEI,levels,Q_CH,
              Q_adjs,Q_small_seps,Q_big_seps,Q_inters,G_list,nI_list,Ok_l3_list,Ik_inc_list,Ik_out_list_list,
              Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,qI,VU,b_norm2(j),sol_pids_bool,sol_qids,x);

    sigma.col(j) = x;
  }
}



void infinite::resolve_fmm_gmres_hybrid(const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const int &num_expansion,
                  const Eigen::VectorXi &qI,
                  const Eigen::SparseMatrix<double> &VU,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const VecXi_list &sing_id_list,
                  const VecXd_list &sing_t_list,
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
                  const std::vector<std::vector<int> > &Q_inters_a,
                  const std::vector<int> &leaf_cells,
                  const Eigen::VectorXi &sub_cells,
                  const Eigen::VectorXi &new_cells,
                  const Eigen::VectorXi &newI,
                  const Eigen::VectorXi &new_pids,
                  const Eigen::VectorXi &sub_qids,
                  const Eigen::VectorXi &new_qids,
                  const std::vector<int> &update_leafs,
                  const std::vector<int> &update_big_seps,
                  const int &nb,
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma)
{

  update_pre_fmm_hybrid(N,L,Q,sing_id_list,sing_t_list,num_expansion,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    sub_cells,new_cells,leaf_cells,Q_adjs,Q_small_seps,Q_big_seps,Q_inters_a,update_leafs,
    update_big_seps,new_pids,sub_qids,new_qids,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list);

  Eigen::VectorXcd Ni = mat2d_to_compvec(N);


  int ns = L.size()/nb;
  Eigen::VectorXi sol_qids;
  std::vector<bool> sol_pids_bool;
  std::vector<bool> con_pids_bool;
  local_resolve_curve(num_expansion,qI,VU,Q_PEI,levels,Q_CH,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,
    newI,new_pids,nb,ns,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,G_list,
    F_list,nI_list,Ok_l3_list,Ik_inc_list,sol_qids,sol_pids_bool,con_pids_bool);


  Eigen::MatrixXd Wf_i, Wf_b, Wg_b;

  Eigen::MatrixXd mu_s = VU*mu;
  Eigen::MatrixXd sigma_s = VU*sigma;

  update_forward_fmm_f(Ni,num_expansion,qI,sol_qids,mu_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,F_list,
    nI_list,Ok_l3_list,Ik_inc_list,sol_pids_bool,Wf_i);

  update_forward_fmm_f(Ni,num_expansion,qI,sol_qids,mu_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,F_list,
    nI_list,Ok_l3_list,Ik_inc_list,con_pids_bool,Wf_b);

  update_forward_fmm_g(num_expansion,qI,sol_qids,sigma_s,leaf_cells,levels,Q_PEI,Q_adjs,Q_small_seps,
    Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,G_list,
    nI_list,Ok_l3_list,Ik_inc_list,con_pids_bool,Wg_b);

  Eigen::MatrixXd b_s = igl::slice(b,sol_qids,1);
  Eigen::MatrixXd bn = b_s - Wf_b - Wf_i - Wg_b;
  Eigen::VectorXd b_norm2 = b_s.colwise().norm();

  int nj = bn.cols();

  if(sigma.rows()<1){
    sigma = Eigen::MatrixXd::Ones(qI.size(),nj);
  }
  for(int j=0; j<nj; j++){
    Eigen::VectorXd x = sigma.col(j);
    
    update_fmm_gmres_hybrid(bn.col(j),restart,tol,maxIters,num_expansion,leaf_cells,Q_PEI,levels,Q_CH,
              Q_adjs,Q_small_seps,Q_big_seps,Q_inters,G_list,nI_list,Ok_l3_list,Ik_inc_list,Ik_out_list_list,
              Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,qI,VU,b_norm2(j),sol_pids_bool,sol_qids,x);

    sigma.col(j) = x;
  }
}


void infinite::solve_fmm_gmres_bem(const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const int &num_expansion,
                  const Eigen::VectorXi &qI,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const VecXi_list &sing_id_list,
                  const VecXd_list &sing_t_list,
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
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma)
{
  Eigen::VectorXcd Ni = mat2d_to_compvec(N);

  pre_fmm(N,L,Q,qI,sing_id_list,sing_t_list,num_expansion,Q_PEI,Q_QI,Q_LL,
    Q_PP,Q_CH,Q_LV,Q_CN,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,
    Ik_child_list,Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list);

  Eigen::MatrixXd Wf;
  forward_fmm_f(Ni,num_expansion,qI,mu,leaf_cells,levels,Q_PEI,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_CH,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,Wf);

  Eigen::MatrixXd bn = b-Wf;
  int nj = bn.cols();

  if(sigma.rows()<1){
    sigma = Eigen::MatrixXd::Ones(qI.size(),nj);
  }
  for(int j=0; j<nj; j++){
    Eigen::VectorXd x = sigma.col(j);

    fmm_gmres(bn.col(j),restart,tol,maxIters,num_expansion,leaf_cells,Q_PEI,levels,Q_CH,
              Q_adjs,Q_small_seps,Q_big_seps,Q_inters,G_list,nI_list,Ok_l3_list,Ik_inc_list,Ik_out_list_list,Ok_inter_list_list,
              Ik_child_list,Ok_inc_list_list,qI,b_norm(j),x);

    sigma.col(j) = x;
  }
}



void infinite::solve_fmm_gmres_bem(const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &C,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixXd &mu,
                  const Eigen::MatrixXd &b,
                  const VecXi_list &sing_id_list,
                  const VecXd_list &sing_t_list,
                  const int &num_expansion,
                  const int &min_pnt_num,
                  const int &max_depth,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  Eigen::MatrixXd &sigma)
{
  // Eigen::MatrixXd mu = u1e-u2e;

  Eigen::MatrixX2d PC(P.rows()+C.rows(),P.cols());
  PC<<P,C;
  
  Eigen::RowVector2d minP = PC.colwise().minCoeff();
  Eigen::RowVector2d maxP = PC.colwise().maxCoeff();


  std::vector<std::vector<int > > Q_PI, Q_PEI;
  Eigen::MatrixXi Q_CH;
  Eigen::VectorXi Q_PA, Q_LV;
  Eigen::MatrixXd Q_CN;
  Eigen::VectorXd Q_W;
  Mat2d_list_list Q_PP;
  RowVec2d_list_list Q_LL;

  std::vector<std::vector<int> > Q_adjs, Q_small_seps, Q_big_seps, Q_inters;


  infinite::quadtree(P,E,C,minP,maxP,max_depth,min_pnt_num,
            Q_PI,Q_PEI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);
  infinite::compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_small_seps,Q_big_seps,Q_inters);

  // Eigen::MatrixXd b = (u1e+u2e)/2;
  Eigen::VectorXd b_norm = b.colwise().norm();


  std::vector<int> leaf_cells;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }
  Eigen::VectorXi qI(C.rows());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    for(int ci=0; ci<Q_PI[leaf_cells[i]].size(); ci++){
      qI(Q_PI[leaf_cells[i]][ci]) = leaf_cells[i];
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  std::vector<VecXcd_list> Ik_out_list_list;
  std::vector<VecXcd_list> Ok_inter_list_list;
  MatXcd_list Ik_child_list;
  std::vector<std::vector<VecXcd_list> > Ok_inc_list_list;
  VecXcd_list lw_list;
  std::vector<RowVecXd_list> G_list;
  std::vector<RowVecXd_list> F_list;
  std::vector<VecXi_list> nI_list;
  std::vector<VecXcd_list> Ok_l3_list;
  VecXcd_list Ik_inc_list;

  sigma = Eigen::MatrixXd::Ones(qI.size(),b.cols());
  solve_fmm_gmres_bem(b,b_norm,mu,restart,tol,maxIters,num_expansion,
    qI,N,L,C,sing_id_list,sing_t_list,Q_PEI,Q_PI,levels,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,sigma);
}

void infinite::solve_fmm_gmres_hybrid(const Eigen::MatrixX2d &P,
                        const Eigen::MatrixX2i &E,
                        const Eigen::MatrixX2d &C,
                        const Eigen::MatrixX2d &N,
                        const Eigen::VectorXd &L,
                        const Eigen::MatrixX2d &Q,
                        const Eigen::MatrixXd &mu,
                        const Eigen::MatrixXd &b,
                        const VecXi_list &sing_id_list,
                        const VecXd_list &sing_t_list,
                        const Eigen::SparseMatrix<double> &VU,
                        const int &num_expansion,
                        const int &min_pnt_num,
                        const int &max_depth,
                        const int &restart,
                        const double &tol,
                        const int &maxIters,
                        Eigen::MatrixXd &sigma)
{
    Eigen::MatrixX2d PQ(P.rows()+Q.rows(),P.cols());
    PQ<<P,Q;
    
    Eigen::RowVector2d minP = PQ.colwise().minCoeff();
    Eigen::RowVector2d maxP = PQ.colwise().maxCoeff();
    
    std::vector<std::vector<int > > Q_PI, Q_PEI, Q_QI;
    // std::vector<std::vector<double> > Q_L;
    Eigen::MatrixXi Q_CH;
    Eigen::VectorXi Q_PA, Q_LV;
    Eigen::MatrixXd Q_CN;
    Eigen::VectorXd Q_W;
    Mat2d_list_list Q_PP;
    RowVec2d_list_list Q_LL;

    std::vector<std::vector<int> > Q_adjs, Q_small_seps, Q_big_seps, Q_inters;

    infinite::quadtree(P,E,C,Q,minP,maxP,max_depth,min_pnt_num,
          Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);

    infinite::compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_small_seps,Q_big_seps,Q_inters);

    std::vector<int> leaf_cells;
    for(int i=0; i<Q_CH.rows(); i++){
      if(Q_CH(i,0)==-1){
        leaf_cells.emplace_back(i);
      }
    }
    Eigen::VectorXi qI(Q.rows());
    #pragma omp parallel for
    for(int i=0; i<leaf_cells.size(); i++){
      for(int ci=0; ci<Q_QI[leaf_cells[i]].size(); ci++){
        qI(Q_QI[leaf_cells[i]][ci]) = leaf_cells[i];
      }
    }
    std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
    for (int i=0; i<Q_LV.size(); i++){
      levels[Q_LV(i)].emplace_back(i);
    }

    Eigen::VectorXd b_norm = b.colwise().norm();
    std::vector<VecXcd_list> Ik_out_list_list;
    std::vector<VecXcd_list> Ok_inter_list_list;
    MatXcd_list Ik_child_list;
    std::vector<std::vector<VecXcd_list> > Ok_inc_list_list;
    VecXcd_list lw_list;
    std::vector<RowVecXd_list> G_list;
    std::vector<RowVecXd_list> F_list;
    std::vector<VecXi_list> nI_list;
    std::vector<VecXcd_list> Ok_l3_list;
    VecXcd_list Ik_inc_list;

    sigma = Eigen::MatrixXd::Ones(qI.size(),b.cols());
    infinite::solve_fmm_gmres_hybrid(b,b_norm,mu,restart,tol,maxIters,num_expansion,
      qI,VU,N,L,Q,sing_id_list,sing_t_list,Q_PEI,Q_QI,levels,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
      Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
      Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,sigma);
}


void infinite::solve_fmm_gmres_hybrid(const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &C,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const Eigen::Matrix2d &minmaxQ,
                  const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const Eigen::VectorXd &xs,
                  const Eigen::VectorXd &xsc,
                  const Eigen::VectorXd &xg,
                  const int &num_expansion,
                  const int &min_pnt_num,
                  const int &max_depth,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  Eigen::VectorXi &qI,
                  std::vector<std::vector<int > > &Q_PI,
                  std::vector<std::vector<int > > &Q_PEI,
                  std::vector<std::vector<int > > &Q_QI,
                  RowVec2d_list_list &Q_LL,
                  Mat2d_list_list &Q_PP,
                  Eigen::VectorXi &Q_PA,
                  Eigen::MatrixXi &Q_CH,
                  Eigen::VectorXi &Q_LV,
                  Eigen::MatrixXd &Q_CN, 
                  Eigen::VectorXd &Q_W,
                  std::vector<std::vector<int> > &Q_adjs,
                  std::vector<std::vector<int> > &Q_small_seps,
                  std::vector<std::vector<int> > &Q_inters,
                  std::vector<std::vector<int> > &Q_big_seps,
                  std::vector<std::vector<int> > &Q_uni_adjs,
                  std::vector<int> &leaf_cells,
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma,
                  std::vector<Eigen::MatrixXd> &sigma_list,
                  std::vector<Eigen::VectorXd> &err_list)
{
  int nb = E.rows()/xsc.size();

  VecXi_list sing_id_list;
  VecXd_list sing_t_list;
  infinite::singular_on_bnd(xs,xg,nb,sing_id_list,sing_t_list);

  Eigen::SparseMatrix<double> VU;
  infinite::legendre_interpolation(xg,xsc,nb,VU);

  Eigen::RowVector2d minP = minmaxQ.row(0);//PQ.colwise().minCoeff();
  Eigen::RowVector2d maxP = minmaxQ.row(1);//PQ.colwise().maxCoeff();

  infinite::quadtree(P,E,C,Q,minP,maxP,max_depth,min_pnt_num,
        Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);

  infinite::compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_small_seps,Q_big_seps,Q_inters,Q_uni_adjs);


  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }
  qI.resize(Q.rows());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    for(int ci=0; ci<Q_QI[leaf_cells[i]].size(); ci++){
      qI(Q_QI[leaf_cells[i]][ci]) = leaf_cells[i];
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  // sigma = Eigen::MatrixXd::Ones(qI.size(),b.cols());
  infinite::solve_fmm_gmres_hybrid(b,b_norm,mu,restart,tol,maxIters,num_expansion,
    qI,VU,N,L,Q,sing_id_list,sing_t_list,Q_PEI,Q_QI,levels,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,sigma,sigma_list,err_list);
}



void infinite::solve_fmm_gmres_hybrid(const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &C,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const Eigen::Matrix2d &minmaxQ,
                  const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const Eigen::VectorXd &xs,
                  const Eigen::VectorXd &xsc,
                  const Eigen::VectorXd &xg,
                  const int &num_expansion,
                  const int &min_pnt_num,
                  const int &max_depth,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  Eigen::VectorXi &qI,
                  std::vector<std::vector<int > > &Q_PI,
                  std::vector<std::vector<int > > &Q_PEI,
                  std::vector<std::vector<int > > &Q_QI,
                  RowVec2d_list_list &Q_LL,
                  Mat2d_list_list &Q_PP,
                  Eigen::VectorXi &Q_PA,
                  Eigen::MatrixXi &Q_CH,
                  Eigen::VectorXi &Q_LV,
                  Eigen::MatrixXd &Q_CN, 
                  Eigen::VectorXd &Q_W,
                  std::vector<std::vector<int> > &Q_adjs,
                  std::vector<std::vector<int> > &Q_small_seps,
                  std::vector<std::vector<int> > &Q_inters,
                  std::vector<std::vector<int> > &Q_big_seps,
                  std::vector<std::vector<int> > &Q_uni_adjs,
                  std::vector<int> &leaf_cells,
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma)
{
  int nb = E.rows()/xsc.size();

  VecXi_list sing_id_list;
  VecXd_list sing_t_list;
  infinite::singular_on_bnd(xs,xg,nb,sing_id_list,sing_t_list);

  Eigen::SparseMatrix<double> VU;
  infinite::legendre_interpolation(xg,xsc,nb,VU);

  // Eigen::MatrixX2d PQ(P.rows()+Q.rows(),P.cols());
  // PQ<<P,Q;

  Eigen::RowVector2d minP = minmaxQ.row(0);//PQ.colwise().minCoeff();
  Eigen::RowVector2d maxP = minmaxQ.row(1);//PQ.colwise().maxCoeff();


  infinite::quadtree(P,E,C,Q,minP,maxP,max_depth,min_pnt_num,
        Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W);

  infinite::compute_cell_list(Q_PA,Q_CH,Q_adjs,Q_small_seps,Q_big_seps,Q_inters,Q_uni_adjs);


  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
    }
  }
  qI.resize(Q.rows());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    for(int ci=0; ci<Q_QI[leaf_cells[i]].size(); ci++){
      qI(Q_QI[leaf_cells[i]][ci]) = leaf_cells[i];
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  // sigma = Eigen::MatrixXd::Ones(qI.size(),b.cols());
  infinite::solve_fmm_gmres_hybrid(b,b_norm,mu,restart,tol,maxIters,num_expansion,
    qI,VU,N,L,Q,sing_id_list,sing_t_list,Q_PEI,Q_QI,levels,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,sigma);
}



void infinite::resolve_fmm_gmres_hybrid(const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &C,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const Eigen::VectorXd &xs,
                  const Eigen::VectorXd &xsc,
                  const Eigen::VectorXd &xg,
                  const int &num_expansion,
                  const int &min_pnt_num,
                  const int &max_depth,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const Eigen::VectorXi &cI,
                  const std::vector<std::vector<int> > &cEI,
                  const Eigen::VectorXi &subI,
                  const double &sub_t,
                  Eigen::VectorXi &qI,
                  std::vector<std::vector<int > > &Q_PI,
                  std::vector<std::vector<int > > &Q_PEI,
                  std::vector<std::vector<int > > &Q_QI,
                  RowVec2d_list_list &Q_LL,
                  Mat2d_list_list &Q_PP,
                  Eigen::VectorXi &Q_PA,
                  Eigen::MatrixXi &Q_CH,
                  Eigen::VectorXi &Q_LV,
                  Eigen::MatrixXd &Q_CN, 
                  Eigen::VectorXd &Q_W,
                  std::vector<std::vector<int> > &Q_adjs,
                  std::vector<std::vector<int> > &Q_small_seps,
                  std::vector<std::vector<int> > &Q_inters,
                  std::vector<std::vector<int> > &Q_big_seps,
                  std::vector<std::vector<int> > &Q_uni_adjs,
                  std::vector<int> &leaf_cells,
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma)
{  
  int ns = xsc.size();
  int ng = xg.size();
  int nb = E.rows()/ns;



  VecXi_list sing_id_list;
  VecXd_list sing_t_list;
  infinite::singular_on_bnd(xs,xg,nb,sing_id_list,sing_t_list);

  Eigen::SparseMatrix<double> VU;
  infinite::legendre_interpolation(xg,xsc,nb,VU);



  Eigen::VectorXi sub_pids(ns*subI.size());
  Eigen::VectorXi sub_qids(ng*subI.size());
  for(int i=0; i<subI.size(); i++){
    for(int j=0; j<ns; j++){
      sub_pids(ns*i+j) = ns*subI(i)+j;
    }
    for(int j=0; j<ng; j++){
      sub_qids(ng*i+j) = ng*subI(i)+j;
    }
  }
  
  Eigen::VectorXi newI = Eigen::VectorXi::LinSpaced(2*subI.size(),nb-2*subI.size(),nb-1);
  Eigen::VectorXi new_pids(ns*newI.size());
  Eigen::VectorXi new_qids(ng*newI.size());
  for(int i=0; i<newI.size(); i++){
    for(int j=0; j<ns; j++){
      new_pids(ns*i+j) = ns*newI(i)+j;
    }
    for(int j=0; j<ng; j++){
      new_qids(ng*i+j) = ng*newI(i)+j;
    }
  }
  Eigen::VectorXi allI = Eigen::VectorXi::LinSpaced(nb,0,nb-1);
  Eigen::VectorXi origI, IA;
  igl::setdiff(allI,subI,origI,IA);

  Eigen::VectorXi orig_qids(origI.size()*ng);
  for(int i=0; i<origI.size(); i++){
    for(int j=0; j<ng; j++){
      orig_qids(ng*i+j) = ng*origI(i)+j;
    }
  }

  int pre_cell_num = Q_PI.size();
  Eigen::VectorXi sub_cells;
  infinite::adap_quadtree(cI,cEI,qI,sub_pids,sub_qids,new_pids,new_qids,P,E,C,Q,
    max_depth,min_pnt_num,Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W,sub_cells);
  int cur_cell_num = Q_PI.size();
  Eigen::VectorXi new_cells = 
  Eigen::VectorXi::LinSpaced(cur_cell_num-pre_cell_num,pre_cell_num,cur_cell_num-1);

  leaf_cells.clear();
  std::set<int> leaf_set;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
      leaf_set.insert(i);
    }
  }
  std::vector<std::vector<int> > Q_inters_a;
  std::vector<int> big_sep_updates;
  infinite::update_cell_list(sub_cells,new_cells,Q_CH,Q_PA,Q_LV,leaf_set,Q_adjs,Q_small_seps,
                      Q_big_seps,Q_inters,Q_uni_adjs,Q_inters_a,big_sep_updates);

  qI.resize(Q.rows());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    for(int ci=0; ci<Q_QI[leaf_cells[i]].size(); ci++){
      qI(Q_QI[leaf_cells[i]][ci]) = leaf_cells[i];
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  update_density(sigma,sub_qids,orig_qids,new_qids,xg,sub_t);

  //

  // sigma = Eigen::MatrixXd::Ones(b.rows(),b.cols());
  infinite::solve_fmm_gmres_hybrid(b,b_norm,mu,restart,tol,maxIters,num_expansion,
    qI,VU,N,L,Q,sing_id_list,sing_t_list,Q_PEI,Q_QI,levels,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,leaf_cells,Ik_out_list_list,Ok_inter_list_list,Ik_child_list,
    Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,Ik_inc_list,sigma);
}







void infinite::resolve_fmm_gmres_hybrid_local(const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &C,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const Eigen::VectorXd &xs,
                  const Eigen::VectorXd &xsc,
                  const Eigen::VectorXd &xg,
                  const int &num_expansion,
                  const int &min_pnt_num,
                  const int &max_depth,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const Eigen::VectorXi &cI,
                  const std::vector<std::vector<int> > &cEI,
                  const Eigen::VectorXi &subI,
                  const Eigen::VectorXi &origI,
                  const double &sub_t,
                  const Eigen::Matrix2d &bndP,
                  Eigen::VectorXi &qI,
                  std::vector<std::vector<int > > &Q_PI,
                  std::vector<std::vector<int > > &Q_PEI,
                  std::vector<std::vector<int > > &Q_QI,
                  RowVec2d_list_list &Q_LL,
                  Mat2d_list_list &Q_PP,
                  Eigen::VectorXi &Q_PA,
                  Eigen::MatrixXi &Q_CH,
                  Eigen::VectorXi &Q_LV,
                  Eigen::MatrixXd &Q_CN, 
                  Eigen::VectorXd &Q_W,
                  std::vector<std::vector<int> > &Q_adjs,
                  std::vector<std::vector<int> > &Q_small_seps,
                  std::vector<std::vector<int> > &Q_inters,
                  std::vector<std::vector<int> > &Q_big_seps,
                  std::vector<std::vector<int> > &Q_uni_adjs,
                  std::vector<int> &leaf_cells,
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma,
                  Eigen::VectorXi &solI,
                  Eigen::VectorXi &newI)
{  
  int ns = xsc.size();
  int ng = xg.size();
  int nb = E.rows()/ns;


  VecXi_list sing_id_list;
  VecXd_list sing_t_list;
  infinite::singular_on_bnd(xs,xg,nb,sing_id_list,sing_t_list);

  Eigen::SparseMatrix<double> VU;
  infinite::legendre_interpolation(xg,xsc,nb,VU);

  Eigen::VectorXi sub_pids(ns*subI.size());
  Eigen::VectorXi sub_qids(ng*subI.size());
  for(int i=0; i<subI.size(); i++){
    for(int j=0; j<ns; j++){
      sub_pids(ns*i+j) = ns*subI(i)+j;
    }
    for(int j=0; j<ng; j++){
      sub_qids(ng*i+j) = ng*subI(i)+j;
    }
  }
  
  newI = Eigen::VectorXi::LinSpaced(2*subI.size(),nb-2*subI.size(),nb-1);
  Eigen::VectorXi new_pids(ns*newI.size());
  Eigen::VectorXi new_qids(ng*newI.size());
  for(int i=0; i<newI.size(); i++){
    for(int j=0; j<ns; j++){
      new_pids(ns*i+j) = ns*newI(i)+j;
    }
    for(int j=0; j<ng; j++){
      new_qids(ng*i+j) = ng*newI(i)+j;
    }
  }
  // Eigen::VectorXi allI = Eigen::VectorXi::LinSpaced(nb,0,nb-1);
  // Eigen::VectorXi origI, IA;
  // igl::setdiff(allI,subI,origI,IA);

  Eigen::VectorXi orig_qids(origI.size()*ng);
  for(int i=0; i<origI.size(); i++){
    for(int j=0; j<ng; j++){
      orig_qids(ng*i+j) = ng*origI(i)+j;
    }
  }
  std::vector<std::vector<int > > pre_Q_PEI=Q_PEI;
  int pre_cell_num = Q_PI.size();
  Eigen::VectorXi sub_cells;
  infinite::adap_quadtree(cI,cEI,qI,sub_pids,sub_qids,new_pids,new_qids,P,E,C,Q,
    max_depth,min_pnt_num,Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W,sub_cells);
  int cur_cell_num = Q_PI.size();
  Eigen::VectorXi new_cells = 
  Eigen::VectorXi::LinSpaced(cur_cell_num-pre_cell_num,pre_cell_num,cur_cell_num-1);

  std::vector<int> pre_leaf_cells = leaf_cells;


  leaf_cells.clear();
  std::set<int> leaf_set;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
      leaf_set.insert(i);
    }
  }
  std::vector<std::vector<int> > Q_inters_a;
  std::vector<int> big_sep_updates;
  infinite::update_cell_list(sub_cells,new_cells,Q_CH,Q_PA,Q_LV,leaf_set,Q_adjs,Q_small_seps,
                      Q_big_seps,Q_inters,Q_uni_adjs,Q_inters_a,big_sep_updates);

  std::vector<int> update_leafs, update_big_seps;
  infinite::determine_update_cells(pre_leaf_cells,leaf_cells,new_cells,new_pids,pre_Q_PEI,Q_PEI,
                    Q_big_seps,big_sep_updates,update_leafs,update_big_seps);


  qI.resize(Q.rows());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    for(int ci=0; ci<Q_QI[leaf_cells[i]].size(); ci++){
      qI(Q_QI[leaf_cells[i]][ci]) = leaf_cells[i];
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  update_density(sigma,sub_qids,orig_qids,new_qids,xg,sub_t);

  std::vector<int> in_list;
  for(int k=0; k<nb; k++){
    for(int i=0; i<ns; i++){
      int ei0 = E(ns*k+i,0);
      int ei1 = E(ns*k+i,1);
      if( (P(ei0,0)>bndP(0,0) && P(ei0,0)<bndP(1,0) && 
      P(ei0,1)>bndP(0,1) && P(ei0,1)<bndP(1,1)) ||
      (P(ei1,0)>bndP(0,0) && P(ei1,0)<bndP(1,0) && 
      P(ei1,1)>bndP(0,1) && P(ei1,1)<bndP(1,1)) )
      {
        in_list.emplace_back(k);
        break;
      }
    }
  }
  Eigen::VectorXi inI = Eigen::Map<Eigen::VectorXi>(in_list.data(),in_list.size());

  infinite::resolve_fmm_gmres_hybrid(b,b_norm,mu,restart,tol,maxIters,num_expansion,
    qI,VU,N,L,Q,sing_id_list,sing_t_list,Q_PEI,Q_QI,levels,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_inters_a,leaf_cells,sub_cells,new_cells,newI,
    new_pids,sub_qids,new_qids,inI,update_leafs,update_big_seps,nb,Ik_out_list_list,
    Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,lw_list,G_list,F_list,nI_list,
    Ok_l3_list,Ik_inc_list,sigma,solI);
}



void infinite::resolve_fmm_gmres_hybrid_local(const Eigen::MatrixX2d &P,
                  const Eigen::MatrixX2i &E,
                  const Eigen::MatrixX2d &C,
                  const Eigen::MatrixX2d &N,
                  const Eigen::VectorXd &L,
                  const Eigen::MatrixX2d &Q,
                  const Eigen::MatrixXd &b, 
                  const Eigen::VectorXd &b_norm,
                  const Eigen::MatrixXd &mu,
                  const Eigen::VectorXd &xs,
                  const Eigen::VectorXd &xsc,
                  const Eigen::VectorXd &xg,
                  const int &num_expansion,
                  const int &min_pnt_num,
                  const int &max_depth,
                  const int &restart,
                  const double &tol,
                  const int &maxIters,
                  const Eigen::VectorXi &cI,
                  const std::vector<std::vector<int> > &cEI,
                  const Eigen::VectorXi &subI,
                  const Eigen::VectorXi &origI,
                  const double &sub_t,
                  const Eigen::Matrix2d &bndP,
                  Eigen::VectorXi &qI,
                  std::vector<std::vector<int > > &Q_PI,
                  std::vector<std::vector<int > > &Q_PEI,
                  std::vector<std::vector<int > > &Q_QI,
                  RowVec2d_list_list &Q_LL,
                  Mat2d_list_list &Q_PP,
                  Eigen::VectorXi &Q_PA,
                  Eigen::MatrixXi &Q_CH,
                  Eigen::VectorXi &Q_LV,
                  Eigen::MatrixXd &Q_CN, 
                  Eigen::VectorXd &Q_W,
                  std::vector<std::vector<int> > &Q_adjs,
                  std::vector<std::vector<int> > &Q_small_seps,
                  std::vector<std::vector<int> > &Q_inters,
                  std::vector<std::vector<int> > &Q_big_seps,
                  std::vector<std::vector<int> > &Q_uni_adjs,
                  std::vector<int> &leaf_cells,
                  std::vector<VecXcd_list> &Ik_out_list_list,
                  std::vector<VecXcd_list> &Ok_inter_list_list,
                  MatXcd_list &Ik_child_list,
                  std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                  VecXcd_list &lw_list,
                  std::vector<RowVecXd_list> &G_list,
                  std::vector<RowVecXd_list> &F_list,
                  std::vector<VecXi_list> &nI_list,
                  std::vector<VecXcd_list> &Ok_l3_list,
                  VecXcd_list &Ik_inc_list,
                  Eigen::MatrixXd &sigma)
{  
  int ns = xsc.size();
  int ng = xg.size();
  int nb = E.rows()/ns;


  VecXi_list sing_id_list;
  VecXd_list sing_t_list;
  infinite::singular_on_bnd(xs,xg,nb,sing_id_list,sing_t_list);

  Eigen::SparseMatrix<double> VU;
  infinite::legendre_interpolation(xg,xsc,nb,VU);

  Eigen::VectorXi sub_pids(ns*subI.size());
  Eigen::VectorXi sub_qids(ng*subI.size());
  for(int i=0; i<subI.size(); i++){
    for(int j=0; j<ns; j++){
      sub_pids(ns*i+j) = ns*subI(i)+j;
    }
    for(int j=0; j<ng; j++){
      sub_qids(ng*i+j) = ng*subI(i)+j;
    }
  }
  
  Eigen::VectorXi newI = Eigen::VectorXi::LinSpaced(2*subI.size(),nb-2*subI.size(),nb-1);
  Eigen::VectorXi new_pids(ns*newI.size());
  Eigen::VectorXi new_qids(ng*newI.size());
  for(int i=0; i<newI.size(); i++){
    for(int j=0; j<ns; j++){
      new_pids(ns*i+j) = ns*newI(i)+j;
    }
    for(int j=0; j<ng; j++){
      new_qids(ng*i+j) = ng*newI(i)+j;
    }
  }
  // Eigen::VectorXi allI = Eigen::VectorXi::LinSpaced(nb,0,nb-1);
  // Eigen::VectorXi origI, IA;
  // igl::setdiff(allI,subI,origI,IA);

  Eigen::VectorXi orig_qids(origI.size()*ng);
  for(int i=0; i<origI.size(); i++){
    for(int j=0; j<ng; j++){
      orig_qids(ng*i+j) = ng*origI(i)+j;
    }
  }
  std::vector<std::vector<int > > pre_Q_PEI=Q_PEI;
  int pre_cell_num = Q_PI.size();
  Eigen::VectorXi sub_cells;
  infinite::adap_quadtree(cI,cEI,qI,sub_pids,sub_qids,new_pids,new_qids,P,E,C,Q,
    max_depth,min_pnt_num,Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_CH,Q_PA,Q_LV,Q_CN,Q_W,sub_cells);
  int cur_cell_num = Q_PI.size();
  Eigen::VectorXi new_cells = 
  Eigen::VectorXi::LinSpaced(cur_cell_num-pre_cell_num,pre_cell_num,cur_cell_num-1);

  std::vector<int> pre_leaf_cells = leaf_cells;


  leaf_cells.clear();
  std::set<int> leaf_set;
  for(int i=0; i<Q_CH.rows(); i++){
    if(Q_CH(i,0)==-1){
      leaf_cells.emplace_back(i);
      leaf_set.insert(i);
    }
  }
  std::vector<std::vector<int> > Q_inters_a;
  std::vector<int> big_sep_updates;
  infinite::update_cell_list(sub_cells,new_cells,Q_CH,Q_PA,Q_LV,leaf_set,Q_adjs,Q_small_seps,
                      Q_big_seps,Q_inters,Q_uni_adjs,Q_inters_a,big_sep_updates);

  std::vector<int> update_leafs, update_big_seps;
  infinite::determine_update_cells(pre_leaf_cells,leaf_cells,new_cells,new_pids,pre_Q_PEI,Q_PEI,
                    Q_big_seps,big_sep_updates,update_leafs,update_big_seps);


  qI.resize(Q.rows());
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    for(int ci=0; ci<Q_QI[leaf_cells[i]].size(); ci++){
      qI(Q_QI[leaf_cells[i]][ci]) = leaf_cells[i];
    }
  }
  std::vector<std::vector<int > > levels(Q_LV.maxCoeff()+1);
  for (int i=0; i<Q_LV.size(); i++){
    levels[Q_LV(i)].emplace_back(i);
  }

  update_density(sigma,sub_qids,orig_qids,new_qids,xg,sub_t);


  infinite::resolve_fmm_gmres_hybrid(b,b_norm,mu,restart,tol,maxIters,num_expansion,
    qI,VU,N,L,Q,sing_id_list,sing_t_list,Q_PEI,Q_QI,levels,Q_LL,Q_PP,Q_CH,Q_LV,Q_CN,
    Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_inters_a,leaf_cells,sub_cells,new_cells,newI,
    new_pids,sub_qids,new_qids,update_leafs,update_big_seps,nb,Ik_out_list_list,
    Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,lw_list,G_list,F_list,nI_list,
    Ok_l3_list,Ik_inc_list,sigma);
}