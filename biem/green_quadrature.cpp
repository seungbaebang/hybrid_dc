#include "green_quadrature.h"


void infinite::green_quadrature(const Eigen::VectorXcd &Pi,
                      const Eigen::VectorXcd &Ni,
                      const Eigen::VectorXd  &L,
                      const std::vector<int> &ids,
                      const std::complex<double> &qi,
                      Eigen::RowVectorXd &G,
                      Eigen::RowVectorXd &F)
{
  double PI_2 = 2*igl::PI;
  G.resize(ids.size());
  F.resize(ids.size());

  for(int j=0; j<ids.size(); j++){
    std::complex<double> pi = Pi(ids[j]);
    std::complex<double> ni = Ni(ids[j]);
    double l = L(ids[j]);

    std::complex<double> ri = pi-qi;
    G(j) = -l*std::real(log(ri))/PI_2;
    F(j) = l*std::real(ni/ri)/PI_2;
  }
}


void infinite::green_quadrature(const Eigen::VectorXcd &Pi,
                      const Eigen::VectorXcd &Ni,
                      const Eigen::VectorXd  &L,
                      const Eigen::VectorXcd &Qi,
                      Eigen::MatrixXd &G,
                      Eigen::MatrixXd &F)
{
  double PI_2 = 2*igl::PI;
  G.resize(Qi.size(),Pi.size());
  F.resize(Qi.size(),Pi.size());

  for(int j=0; j<Pi.size(); j++){
    std::complex<double> pi = Pi(j);
    std::complex<double> ni = Ni(j);
    double l = L(j);
    for(int i=0; i<Qi.size(); i++){
      std::complex<double> qi = Qi(i);
      std::complex<double> ri = pi-qi;
      G(i,j) = -l*std::real(log(ri))/PI_2;
      F(i,j) = l*std::real(ni/ri)/PI_2;
    }
  }
}