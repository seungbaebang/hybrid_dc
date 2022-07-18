#include "Ik.h"


std::complex<double> infinite::compute_Ik(const std::complex<double> &z, const int &k)
{
  if(k<=0)
    return 1.0;
  else
    return std::pow(z,k)/(double)(Eigen::VectorXi::LinSpaced(k,1,k).prod());
}

void infinite::compute_Ik(const std::complex<double> &z, const int &np, Eigen::MatrixXcd& Ik, const int &j)
{
  Ik(j,0)=1;
  Ik(j,1)=z;
  for(int k=2; k<np; k++){
    Ik(j,k) = Ik(j,k-1)*(z/(double)k);
  }
}

void infinite::compute_Ik(const std::complex<double> &z, const int &np, Eigen::VectorXcd& Ik)
{
  Ik(0)=1;
  Ik(1)=z;
  for(int k=2; k<np; k++){
    Ik(k) = Ik(k-1)*(z/(double)k);
  }
}

std::complex<double> infinite::compute_Ok(const std::complex<double> &z, const int &k)
{
  if(k<=0)
    return -log(z);
  else
    return ((double)Eigen::VectorXi::LinSpaced(k-1,1,k-1).prod())/std::pow(z,k);
}


void infinite::compute_Ok(const std::complex<double> &z, const int &k, std::complex<double> &v)
{
  if(k<=0)
    v = -log(z);
  else
    v = ((double)Eigen::VectorXi::LinSpaced(k-1,1,k-1).prod())/std::pow(z,k);
}

void infinite::compute_Ok(const std::complex<double> &z, const int &np, Eigen::VectorXcd& Ok)
{
  std::complex<double> zm = z;
  if(std::abs(z) < 1e-12){
    zm = std::complex<double>(1e-12,0);
  }
  Ok(0) = -log(zm);
  Ok(1) = 1.0/zm;
  for(int k=2; k<np; k++){
    Ok(k) = Ok(k-1)*((double)(k-1)/zm);
  }
}

void infinite::compute_Ok_no_log(const std::complex<double> &z, const int &np, Eigen::VectorXcd& Ok)
{
  std::complex<double> zm = z;
  if(std::abs(z) < 1e-12){
    zm = std::complex<double>(1e-12,0);
  }
  Ok(0) = 1.0/zm;
  for(int k=1; k<np; k++){
    Ok(k) = Ok(k-1)*((double)k/zm);
  }
}

void infinite::compute_Ok(const std::complex<double> &z, const int &np, Eigen::MatrixXcd& Ok, const int &j)
{
  std::complex<double> zm = z;
  if(std::abs(z) < 1e-12){
    zm = std::complex<double>(1e-12,0);
  }
  Ok(j,0) = -log(zm);
  Ok(j,1) = 1.0/zm;
  for(int k=2; k<np; k++){
    Ok(j,k) = Ok(j,k-1)*((double)(k-1)/zm);
  }
}