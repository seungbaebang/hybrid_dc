#include "bezier_length.h"

void infinite::bezier_length(const Mat42_list &CPs, 
                    const int &nl,
                    Eigen::VectorXd &L)
{
  int nb = CPs.size();
  Eigen::VectorXd T,W;
  infinite::legendre_gauss_quadrature(nl,0,1,T,W);

  L.resize(nb);
  for(int k=0; k<nb; k++){  
    Eigen::MatrixX2d dP = derivative_bezier(CPs[k],T);
    L(k)=(W.array()*dP.rowwise().norm().array()).sum();
  }

}

void infinite::bezier_length(const Mat42_list &CPs, 
                   const int &ng,
                   const int &nl,
                   Eigen::VectorXd & wL) 
{
  int nb = CPs.size();
  Eigen::VectorXd T,W;
  infinite::legendre_gauss_quadrature(nl,0,1,T,W);

  Eigen::MatrixX2d N;

  bezier_curves_normals(CPs,T,infinite::bezier,N);

  Eigen::VectorXd aL = W.replicate(nb,1).array()*N.rowwise().norm().array();

  wL.resize(nb*ng,1);

  for(int k=0; k<nb; k++){
    //wL.segment(ng*k,ng) = aL.segment(nl*k,nl).sum().replicate(ng,1);
    wL.segment(ng*k,ng) = aL.segment(nl*k,nl).sum()*Eigen::VectorXd::Ones(ng);
  }
}
Eigen::VectorXd infinite::bezier_length(const Mat42_list &CPs)
{
  Eigen::VectorXd L;
  bezier_length(CPs,20,L);
  return L;
}


Eigen::VectorXd infinite::bezier_length(const Mat42_list &CPs, 
                   const int &ng)
{
  Eigen::VectorXd wL;
  bezier_length(CPs,ng,20,wL);
  return wL;
}

Eigen::VectorXd infinite::cubic_length(const Mat42 &CP, 
                                      const Eigen::MatrixX2d &TE, 
                                      const int &ng)
{
  Eigen::VectorXd L(TE.rows());

  for(int k=0; k<TE.rows(); k++){
    Eigen::VectorXd Tk,Wk;
    infinite::legendre_gauss_quadrature(ng,TE(k,0),TE(k,1),Tk,Wk);
    Eigen::MatrixX2d dP = derivative_bezier(CP,Tk);
    L(k)=(Wk.array()*dP.rowwise().norm().array()).sum();
  }
}

Eigen::VectorXd infinite::cubic_length(const Mat42 &CP, 
                                      const Eigen::VectorXd &st,
                                      const Eigen::VectorXd &et,
                                      const int &ng)
{
  Eigen::VectorXd L(st.size());

  for(int k=0; k<st.size(); k++){
    Eigen::VectorXd Tk,Wk;
    infinite::legendre_gauss_quadrature(ng,st(k),et(k),Tk,Wk);
    Eigen::MatrixX2d dP = derivative_bezier(CP,Tk);
    L(k)=(Wk.array()*dP.rowwise().norm().array()).sum();
  }
  return L;
}

Eigen::VectorXd infinite::cubic_length(const Mat42 &CP, 
                                      const Eigen::VectorXd &st,
                                      const Eigen::VectorXd &et)
{
  return cubic_length(CP,st,et,20);
}