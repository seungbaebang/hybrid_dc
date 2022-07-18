
#include "interpolate_density.h"



Eigen::MatrixXd infinite::interpolate_density(const Eigen::VectorXd& xg,
                                    const VecXd_list& xe_list,
                                    const Eigen::MatrixXd& sigma_g)
{
  int ng = xg.size();
  int nb = xe_list.size();
  int nc = sigma_g.cols();
  int np = 0;
  for(int k=0; k<xe_list.size(); k++)
    np=np+xe_list[k].size();

  Eigen::VectorXd xg0 = 2*xg.array()-1;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ng,ng);
  Eigen::MatrixXd Pg = legendre_transform(xg0,ng);
  Eigen::MatrixXd Pg_inv = Pg.householderQr().solve(I);

  Eigen::MatrixXd sigma_e = Eigen::MatrixXd::Zero(np,nc);
  np=0;
  for(int k=0; k<xe_list.size(); k++){
    Eigen::VectorXd xe = xe_list[k];
    int ne = xe.size();
    Eigen::VectorXd xe0 = 2*xe.array()-1;
    Eigen::MatrixXd Pe = legendre_transform(xe0,ng);
    Eigen::MatrixXd T = Pe*Pg_inv;
    sigma_e.block(np,0,ne,nc) = T*sigma_g.block(ng*k,0,ng,nc);
    np=np+ne;
  }
  return sigma_e;
}


Eigen::MatrixXd infinite::interpolate_density(const Eigen::VectorXd& xg,
                                    const Eigen::VectorXd& xe,
                                    const Eigen::MatrixXd& sigma_s)
{
  int ng = xg.size();
  int ne = xe.size();
  int nb = sigma_s.rows()/ng;

  Eigen::VectorXd xg0 = 2*xg.array()-1;
  Eigen::VectorXd xe0 = 2*xe.array()-1;

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ng,ng);

  Eigen::MatrixXd Pg = legendre_transform(xg0,ng);
  Eigen::MatrixXd Pg_inv = Pg.householderQr().solve(I);

  Eigen::MatrixXd Pe = legendre_transform(xe0,ng);

  Eigen::MatrixXd T = Pe*Pg_inv;

  int nc = sigma_s.cols();

  Eigen::MatrixXd sigma_e = Eigen::MatrixXd::Zero(ne*nb,nc);
  for(int i=0; i<nb; i++){
    sigma_e.block(ne*i,0,ne,nc)=T*sigma_s.block(ng*i,0,ng,nc);
  }

  return sigma_e;
}





Eigen::VectorXd infinite::interpolate_density(const Eigen::VectorXd& xg,
                                    const Eigen::VectorXd& xe,
                                    const Eigen::VectorXd& sigma_s)
{
  int ng = xg.size();
  int ne = xe.size();
  int nb = sigma_s.rows()/ng;

  Eigen::VectorXd xg0 = 2*xg.array()-1;
  Eigen::VectorXd xe0 = 2*xe.array()-1;

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ng,ng);

  Eigen::MatrixXd Pg = legendre_transform(xg0,ng);
  Eigen::MatrixXd Pg_inv = Pg.householderQr().solve(I);

  Eigen::MatrixXd Pe = legendre_transform(xe0,ng);

  Eigen::MatrixXd T = Pe*Pg_inv;

  Eigen::VectorXd sigma_e = Eigen::VectorXd::Zero(ne*nb);
  for(int i=0; i<nb; i++){
    sigma_e.segment(ne*i,ne)=T*sigma_s.segment(ng*i,ng);
  }

  return sigma_e;
}

double infinite::interpolate_density(const Eigen::VectorXd& xg,
                                    const double& t,
                                    const Eigen::VectorXd& sigma_s)
{
  int ng = xg.size();

  Eigen::VectorXd xg0 = 2*xg.array()-1;
  double t0 = 2*t-1;

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ng,ng);

  Eigen::MatrixXd Pg = legendre_transform(xg0,ng);
  Eigen::MatrixXd Pg_inv = Pg.householderQr().solve(I);

  Eigen::RowVectorXd Pe = legendre_transform(t0,ng);

  return Pe*Pg_inv*sigma_s;
}

void infinite::update_density(const Eigen::MatrixXd &sigma,
                              const Eigen::VectorXi &sub_ids,
                              const Eigen::VectorXi &orig_ids,
                              const Eigen::VectorXi &new_sub_ids,
                              const Eigen::VectorXd &xg,
                              const double &sub_t,
                              Eigen::MatrixXd &sigma_n)
{
  int ng = xg.size();
  Eigen::MatrixXd sigma_s;
  igl::slice(sigma,sub_ids,1,sigma_s);

  Eigen::VectorXd xg_s(2*ng);
  xg_s.segment(0,ng) = xg*sub_t;
  xg_s.segment(ng-1,ng) = xg.array()*(1.0-sub_t)+sub_t;

  sigma_n.resize(orig_ids.size()+2*sub_ids.size(),sigma.cols());

  Eigen::VectorXi new_orig_ids = Eigen::VectorXi::LinSpaced(orig_ids.size(),0,orig_ids.size()-1);
  igl::slice_into(igl::slice(sigma,orig_ids,1),new_orig_ids,1,sigma_n);
  igl::slice_into(infinite::interpolate_density(xg,xg_s,sigma_s),new_sub_ids,1,sigma_n);
}

void infinite::update_density(Eigen::MatrixXd &sigma,
                              const Eigen::VectorXi &sub_ids,
                              const Eigen::VectorXi &orig_ids,
                              const Eigen::VectorXi &new_sub_ids,
                              const Eigen::VectorXd &xg,
                              const double &sub_t)
{
  Eigen::MatrixXd sigma_n;
  update_density(sigma,sub_ids,orig_ids,new_sub_ids,xg,sub_t,sigma_n);
  sigma = sigma_n;
}