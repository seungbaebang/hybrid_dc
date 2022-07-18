#include "bezier_arclen_param.h"



// void infinite::bezier_arclen_param_old(const Mat42 &CP, 
//                          const Eigen::VectorXd &T,
//                          Eigen::VectorXd& TN) 
// {
//   TN=T;
//   Eigen::VectorXd X,W;
//   int ng=30;
//   infinite::legendre_gauss_quadrature(ng,0,1,X,W);

//   Eigen::MatrixX2d dP = derivative_bezier(CP,X);
//   double la = (W.array()*dP.rowwise().norm().array()).sum();
//   ng=12;
//   infinite::legendre_gauss_quadrature(ng,0,1,X,W);

//   int nt = T.size();
//   for(int i=0; i<nt; i++){
//     double tn0, t0;
//     if(i==0){
//       tn0=0; t0=0;}
//     else{
//       tn0 = TN(i-1); t0 = T(i-1);}
//     double tn1=TN(i);
//     for(int ii=0; ii<30; ii++){
//       Eigen::VectorXd tv = (tn1-tn0)*X.array()+tn0;
//       dP = derivative_bezier(CP,tv);
//       double l = (tn1-tn0)*(W.array()*dP.rowwise().norm().array()).sum()/la;
//       double d = l-(T(i)-t0);
//       tn1 = tn1-d/3;
//     }
//     TN(i)=tn1;
//   }
// }


void infinite::bezier_arclen_param(const Mat42 &CP, 
                         const Eigen::VectorXd &T,
                         Eigen::VectorXd& TN) 
{
  TN=T;

  // std::vector<int> end_ids;
  // std::vector<int> sub_ids;
  // for(int k=0; k<T.size(); k++){
  //   if(T(k)==0 || T(k)==1)
  //     end_ids.emplace_back(k);
  //   else
  //     sub_ids.emplace_back(k);
  // }
  // if(sub_ids.size()==0)
  //   return;


  Eigen::VectorXd X,W;
  int ng=30;
  infinite::legendre_gauss_quadrature(ng,0,1,X,W);

  Eigen::VectorXd L(ng);
  for(int k=0; k<ng; k++){
    Eigen::VectorXd Xk,Wk;
    infinite::legendre_gauss_quadrature(ng,0,X(k),Xk,Wk);
    Eigen::MatrixX2d dPk = derivative_bezier(CP,Xk);
    L(k) = (Wk.array()*dPk.rowwise().norm().array()).sum();
  }

  Eigen::MatrixX2d dP = derivative_bezier(CP,X);
  double la = (W.array()*dP.rowwise().norm().array()).sum();

  Eigen::VectorXd Tc=T;

  for(int k=0; k<30; k++){
    Eigen::VectorXd Ls = infinite::interpolate_density(X,Tc,L);
    Eigen::VectorXd D = Ls/la-T;
    if(D.array().abs().maxCoeff()<1e-5)
      break;
    Eigen::MatrixX2d dP = derivative_bezier(CP,Tc);
    Eigen::VectorXd dD = dP.rowwise().norm()/la;
    for(int i=0; i<dD.size(); i++){
      if(dD(i)==0)
        dD(i)=1e-20;
    }
    TN = Tc - Eigen::VectorXd(D.array()/dD.array());
    for(int i=0; i<TN.size(); i++){
      if(TN(i)>1)
        TN(i)=1;
      if(TN(i)<0)
        TN(i)=0;
    }
    Tc = TN;
  }
}



void infinite::bezier_arclen_param(const Mat42 &CP, 
                                  const double &t,
                                  double &tn) 
{
  tn=t;

  Eigen::VectorXd X,W;
  int ng=30;
  infinite::legendre_gauss_quadrature(ng,0,1,X,W);

  Eigen::VectorXd L(ng);
  for(int k=0; k<ng; k++){
    Eigen::VectorXd Xk,Wk;
    infinite::legendre_gauss_quadrature(ng,0,X(k),Xk,Wk);
    Eigen::MatrixX2d dPk = derivative_bezier(CP,Xk);
    L(k) = (Wk.array()*dPk.rowwise().norm().array()).sum();
  }

  Eigen::MatrixX2d dP = derivative_bezier(CP,X);
  double la = (W.array()*dP.rowwise().norm().array()).sum();

  double tc=t;

  for(int k=0; k<30; k++){
    double ls = infinite::interpolate_density(X,tc,L);
    double d = ls/la-t;
    if(std::abs(d)<1e-5)
      break;
    Eigen::RowVector2d dP = derivative_bezier(CP,tc);
    double dD = dP.norm()/la;
    if(dD==0)
      dD=1e-20;
    tn = tc - d/dD;;
    if(tn>1)
      tn=1;
    if(tn<0)
      tn=0;
    tc = tn;
  }
}