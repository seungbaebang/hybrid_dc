
#include "legendre_transform.h"

Eigen::RowVectorXd infinite::legendre_transform(const double &t, const int &n)
{
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n,n);
  Eigen::RowVectorXd X = Eigen::RowVectorXd::Zero(n);
  C(0,0)=1; C(1,1)=1;

  X(0)=1;
  X.segment(1,n-1)=t*Eigen::RowVectorXd::Ones(n-1);


  Eigen::VectorXi mI(n);
  mI.segment(1,n-1) = Eigen::ArrayXi::LinSpaced(n-1,0,n-2);
  mI(0)=n-1;

  for(int i=2; i<n; i++){
    Eigen::MatrixXd Cm = igl::slice(C,mI,2);
    C.row(i)=(2.0-1.0/(double)i)*Cm.row(i-1) - (1.0-1.0/(double)i)*C.row(i-2);
    X(i)=X(i)*X(i-1);
  }

  Eigen::RowVectorXd V = X*C.transpose();

  return V;
}

Eigen::MatrixXd infinite::legendre_transform(const Eigen::VectorXd &T, const int &n)
{
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n,n);
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(T.size(),n);
  C(0,0)=1; C(1,1)=1;

  X.col(0)=Eigen::VectorXd::Ones(T.size());
  X.block(0,1,T.size(),n-1)=T.replicate(1,n-1);

  Eigen::VectorXi mI(n);
  mI.segment(1,n-1) = Eigen::ArrayXi::LinSpaced(n-1,0,n-2);
  mI(0)=n-1;


  for(int i=2; i<n; i++){
    Eigen::MatrixXd Cm = igl::slice(C,mI,2);
    C.row(i)=(2.0-1.0/(double)i)*Cm.row(i-1) - (1.0-1.0/(double)i)*C.row(i-2);
    X.col(i)=X.col(i).array()*X.col(i-1).array();
  }

  Eigen::MatrixXd V = X*C.transpose();

  return V;
}

Eigen::MatrixXd infinite::legendre_interpolation(const Eigen::VectorXd &xg, 
                                       const Eigen::VectorXd &xc)
{
  int ng = xg.size();
  int nc = xc.size();

  Eigen::VectorXd xg0 = 2*xg.array()-1;
  Eigen::VectorXd xc0 = 2*xc.array()-1;

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ng,ng);

  Eigen::MatrixXd Pg = legendre_transform(xg0,ng);
  Eigen::MatrixXd Pg_inv = Pg.householderQr().solve(I);

  Eigen::MatrixXd Pc = legendre_transform(xc0,ng);

  Eigen::MatrixXd T = Pc*Pg_inv;
  return T;
}

void infinite::legendre_interpolation(const Eigen::VectorXd &xg, 
                      const Eigen::VectorXd &xc,
                      const int &nb,
                      Eigen::SparseMatrix<double> &ST)
{
  Eigen::MatrixXd T = legendre_interpolation(xg,xc);

  typedef Eigen::Triplet<double> Trip;

  std::vector<Trip> coeffs(nb*T.size());
  int ti=0;
  for(int k=0; k<nb; k++){
    for(int ri=0; ri<T.rows(); ri++){
      for(int ci=0; ci<T.cols(); ci++){
        coeffs[ti++] = Trip(k*T.rows()+ri,k*T.cols()+ci,T(ri,ci));
      }
    }
  }

  ST.resize(nb*T.rows(), nb*T.cols());

  ST.setFromTriplets(coeffs.begin(), coeffs.end());

}

// #ifdef INFINITE_STATIC_LIBRARY
// template Eigen::Matrix<double, -1, -1, 0, -1, -1> igl::slice<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::DenseBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int);
// #endif