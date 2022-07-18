
#include "legendre_gauss_quadrature.h"

void infinite::legendre_gauss_quadrature(const int &ng, 
            const double &a, const double &b,
            Eigen::VectorXd &X, Eigen::VectorXd &W)
{
  double eps = 2.2204e-16;
  int n=ng-1, n1=n+1, n2=n+2;
  Eigen::VectorXd xu = Eigen::VectorXd::LinSpaced(n1,-1,1);

  Eigen::VectorXd lx = Eigen::VectorXd::LinSpaced(n1,0,n);

  Eigen::VectorXd y=cos((2*lx.array()+1)*igl::PI/(2*n+2)) 
                    + (0.27/n1)*sin(igl::PI*xu.array()*n/n2);

  Eigen::VectorXd y0=2*Eigen::VectorXd::Ones(n1);

  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n1,n2);
  Eigen::VectorXd Lp;

  while((y-y0).cwiseAbs().maxCoeff()>eps){
    L.col(0)=Eigen::VectorXd::Ones(n1);
    L.col(1)=y;
    for(int k=1; k<n1; k++){
      double dk = (double)k;
      L.col(k+1) = ( (2*dk+1)*y.array()*L.col(k).array()-dk*L.col(k-1).array() )/(dk+1);
    }
    Lp = n2*( L.col(n1-1).array()-y.array()*L.col(n2-1).array() )/(1-y.array().square());
    y0=y;
    y=y0.array()-L.col(n2-1).array()/Lp.array();
  }
  X = (a*(1.0-y.array())+b*(1.0+y.array()))/2;
  double dn = (double)n2/(double)n1;
  W = (b-a)/( ((1.0-y.array().square())*Lp.array().square()) )*dn*dn ;
  X=X.colwise().reverse().eval();
  W=W.colwise().reverse().eval();
}