#include "subdivide_curves.h"

void infinite::cubic_split(const Mat42 &C,double t,Mat42 &C1,Mat42 &C2)
{
  Eigen::RowVector2d C12 = (C.row(1)-C.row(0))*t + C.row(0);
  Eigen::RowVector2d C23 = (C.row(2)-C.row(1))*t + C.row(1);
  Eigen::RowVector2d C34 = (C.row(3)-C.row(2))*t + C.row(2);

  Eigen::RowVector2d C123 = (C23-C12)*t+C12;
  Eigen::RowVector2d C234 = (C34-C23)*t+C23;
  Eigen::RowVector2d C1234 = (C234-C123)*t+C123;
  C1.row(0)=C.row(0);
  C1.row(1)=C12;
  C1.row(2)=C123;
  C1.row(3)=C1234;
  C2.row(0)=C1234;
  C2.row(1)=C234;
  C2.row(2)=C34;
  C2.row(3)=C.row(3);
}

void infinite::split_diffcurv_color(
                    const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const double &t,
                    Eigen::MatrixX3d &C1,
                    Eigen::VectorXd &X1,
                    Eigen::MatrixX3d &C2,
                    Eigen::VectorXd &X2)
{
  Eigen::Matrix3d CN;
  CN.row(0)=C.row(0);
  CN.row(2)=C.row(C.rows()-1);
  CN.row(1)=eval_diffcurv_color(C,X,t);

  std::vector<Eigen::RowVector3d> c1_list;
  std::vector<Eigen::RowVector3d> c2_list;
  std::vector<double> x1_list;
  std::vector<double> x2_list;


  c1_list.emplace_back(CN.row(0));
  c2_list.emplace_back(CN.row(1));

  for(int k=0; k<X.size(); k++){
    if(X(k)<=t && X(k)>0){
      c1_list.emplace_back(C.row(k+1));
      x1_list.emplace_back(X(k)/t);
    }
    else if(X(k)>t && X(k)<1){
      c2_list.emplace_back(C.row(k+1));
      x2_list.emplace_back((X(k)-t)/(1.0-t));
    }
  }
  c1_list.emplace_back(CN.row(1));
  c2_list.emplace_back(CN.row(2));

  C1.resize(c1_list.size(),3);
  C2.resize(c2_list.size(),3);
  X1.resize(x1_list.size());
  X2.resize(x2_list.size());

  for(int k=0; k<c1_list.size(); k++)
    C1.row(k)=c1_list[k];
  for(int k=0; k<c2_list.size(); k++)
    C2.row(k)=c2_list[k];
  for(int k=0; k<x1_list.size(); k++)
    X1(k)=x1_list[k];
  for(int k=0; k<x2_list.size(); k++)
    X2(k)=x2_list[k];
}

void infinite::subdivide_diffcurv(
                const Mat42_list &CPs,
                const MatX3d_list &CLCs,
                const VecXd_list &CLTs,
                const MatX3d_list &CRCs,
                const VecXd_list &CRTs,
                const double &t,
                Mat42_list &nCPs,
                MatX3d_list &nCLCs,
                VecXd_list &nCLTs,
                MatX3d_list &nCRCs,
                VecXd_list &nCRTs)
{
  nCPs.resize(2*CPs.size());
  nCLCs.resize(2*CLCs.size());
  nCLTs.resize(2*CLTs.size());
  nCRCs.resize(2*CRCs.size());
  nCRTs.resize(2*CRTs.size());

  for(int k=0; k<CPs.size(); k++){
    Mat42 CP = CPs[k];
    Eigen::MatrixX3d CLC = CLCs[k];
    Eigen::VectorXd CLT = CLTs[k];
    Eigen::MatrixX3d CRC = CRCs[k];
    Eigen::VectorXd CRT = CRTs[k];
    Mat42 CP1, CP2;
    Eigen::MatrixX3d CLC1, CLC2, CRC1, CRC2;
    Eigen::VectorXd CLT1, CLT2, CRT1, CRT2;

    double tn;
    bezier_arclen_param(CP,t,tn);

    cubic_split(CP,tn, CP1,CP2);

    split_diffcurv_color(CLC,CLT,t,CLC1,CLT1,CLC2,CLT2);
    split_diffcurv_color(CRC,CRT,t,CRC1,CRT1,CRC2,CRT2);

    nCPs[2*k]=CP1; nCPs[2*k+1]=CP2;
    nCLCs[2*k]=CLC1; nCLCs[2*k+1]=CLC2;
    nCLTs[2*k]=CLT1; nCLTs[2*k+1]=CLT2;
    nCRCs[2*k]=CRC1; nCRCs[2*k+1]=CRC2;
    nCRTs[2*k]=CRT1; nCRTs[2*k+1]=CRT2;
  }
}

void infinite::subdivide_diffcurv(
                Mat42_list &CPs,
                MatX3d_list &CLCs,
                VecXd_list &CLTs,
                MatX3d_list &CRCs,
                VecXd_list &CRTs,
                const double &t)
{
  Mat42_list nCPs;
  MatX3d_list nCLCs;
  VecXd_list nCLTs;
  MatX3d_list nCRCs;
  VecXd_list nCRTs;
  subdivide_diffcurv(CPs,CLCs,CLTs,CRCs,CRTs,t,
                        nCPs,nCLCs,nCLTs,nCRCs,nCRTs);
  CPs=nCPs;CLCs=nCLCs;CLTs=nCLTs;CRCs=nCRCs;CRTs=nCRTs;
}


Eigen::VectorXi infinite::get_adap_subdivision_list(
                const Eigen::VectorXd& L,
                const Eigen::VectorXd& xg,
                const Eigen::MatrixXd &sigma,
                const double &pixel_length)
{
  std::vector<int> sub_list;
  int nb = L.size();
  int ng = xg.size();
  Eigen::VectorXd xg0 = 2*xg.array()-1;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ng,ng);

  Eigen::MatrixXd Pg = legendre_transform(xg0,ng);
  Eigen::MatrixXd Pg_inv = Pg.householderQr().solve(I);


  for(int k=0; k<nb; k++){
    if(L(k)<4*pixel_length){
      continue;
    }
    Eigen::MatrixXd legendre_coeff = Pg_inv*sigma.block(k*ng,0,ng,sigma.cols());
    double llc = L(k)*legendre_coeff.row(ng-1).cwiseAbs().maxCoeff();


    if(llc>1.5){
      sub_list.emplace_back(k);
    }

  }
  Eigen::VectorXi subI = Eigen::Map<Eigen::VectorXi>(sub_list.data(),sub_list.size());
  return subI;
}