#include "diffcurv.h"


void collect_color(const Eigen::MatrixX3d &C,
                  const Eigen::VectorXd &T,
                  const Eigen::MatrixX3d &Cn,
                  const int &k,
                  Eigen::MatrixX3d &CC,
                  Eigen::VectorXd &CT)
{
  std::vector<Eigen::RowVector3d> color_list(1);
  std::vector<double> t_list;
  color_list[0] = Cn.row(k);
  for(int j=0; j<T.size(); j++){
    if(T(j)>(double)k && T(j)<(double)(k+1)){
      color_list.emplace_back(C.row(j));
      double tt = T(j)-(double)k;
      t_list.emplace_back(tt);
    }
  }
  color_list.emplace_back(Cn.row(k+1));
  CC.resize(color_list.size(),3);
  for(int j=0; j<color_list.size(); j++){
    CC.row(j)=color_list[j];
  }
  CT.resize(t_list.size());
  for(int j=0; j<t_list.size(); j++){
    CT(j) = t_list[j];
  }

}


void infinite::segment_diffcurv_color(const Eigen::VectorXi &dcn,
                                const MatX3d_list &CLCl,
                                const VecXd_list &CLTl,
                                const MatX3d_list &CRCl,
                                const VecXd_list &CRTl,
                                MatX3d_list &CLCs,
                                VecXd_list &CLTs,
                                MatX3d_list &CRCs,
                                VecXd_list &CRTs)
{


  CLCs.clear(); CLTs.clear(); CRCs.clear(); CRTs.clear(); 
  for(int i=0; i<CLCl.size(); i++){
    int nb = dcn(i);
    Eigen::MatrixX3d CLC = CLCl[i];
    Eigen::VectorXd CLT = CLTl[i];
    Eigen::MatrixX3d CRC = CRCl[i];
    Eigen::VectorXd CRT = CRTl[i];

    Eigen::MatrixX3d CLCn(nb+1,3);
    Eigen::MatrixX3d CRCn(nb+1,3);
    for(int k=0; k<(nb+1); k++){
      CLCn.row(k) = query_color(CLC,CLT,(double)k);
      CRCn.row(k) = query_color(CRC,CRT,(double)k);
    }
    for(int k=0; k<nb; k++){
      Eigen::MatrixX3d sCLC, sCRC;
      Eigen::VectorXd sCLT, sCRT;

      collect_color(CLC,CLT,CLCn,k,sCLC,sCLT);
      collect_color(CRC,CRT,CRCn,k,sCRC,sCRT);

      CLCs.emplace_back(sCLC);
      CLTs.emplace_back(sCLT);
      CRCs.emplace_back(sCRC);
      CRTs.emplace_back(sCRT);
    }
  }
}




void infinite::segment_diffcurv(const Eigen::MatrixX2d &CP,
                                const Eigen::MatrixX4i &CE,
                                const Eigen::VectorXi &CI,
                                const MatX3d_list &CLCl,
                                const VecXd_list &CLTl,
                                const MatX3d_list &CRCl,
                                const VecXd_list &CRTl,
                                Mat42_list &CPs,
                                MatX3d_list &CLCs,
                                VecXd_list &CLTs,
                                MatX3d_list &CRCs,
                                VecXd_list &CRTs)
{
  Eigen::VectorXi dcn = Eigen::VectorXi::Zero(CI.maxCoeff()+1);
  for(int i=0; i<CI.size(); i++){
    dcn(CI(i)) += 1;
  }
  
  CPs.resize(CE.rows());
  for(int i=0; i<CE.rows(); i++){
    Mat42 sCP;
    sCP.row(0) = CP.row(CE(i,0));
    sCP.row(1) = CP.row(CE(i,1));
    sCP.row(2) = CP.row(CE(i,2));
    sCP.row(3) = CP.row(CE(i,3));
    CPs[i]=sCP;
  }

  CLCs.clear(); CLTs.clear(); CRCs.clear(); CRTs.clear(); 
  for(int i=0; i<CLCl.size(); i++){
    int nb = dcn(i);
    Eigen::MatrixX3d CLC = CLCl[i];
    Eigen::VectorXd CLT = CLTl[i];
    Eigen::MatrixX3d CRC = CRCl[i];
    Eigen::VectorXd CRT = CRTl[i];

    Eigen::MatrixX3d CLCn(nb+1,3);
    Eigen::MatrixX3d CRCn(nb+1,3);
    for(int k=0; k<(nb+1); k++){
      CLCn.row(k) = query_color(CLC,CLT,(double)k);
      CRCn.row(k) = query_color(CRC,CRT,(double)k);
    }

    for(int k=0; k<nb; k++){
      Eigen::MatrixX3d sCLC, sCRC;
      Eigen::VectorXd sCLT, sCRT;

      collect_color(CLC,CLT,CLCn,k,sCLC,sCLT);
      collect_color(CRC,CRT,CRCn,k,sCRC,sCRT);

      CLCs.emplace_back(sCLC);
      CLTs.emplace_back(sCLT);
      CRCs.emplace_back(sCRC);
      CRTs.emplace_back(sCRT);

      // std::vector<Eigen::RowVector3d> l_color_list(1);
      // std::vector<double> l_t_list;
      // l_color_list[0] = CLCn.row(k);
      // for(int j=0; j<CLT.size(); j++){
      //   if(CLT(j)>(double)k && CLT(j)<(double)(k+1)){
      //     l_color_list.emplace_back(CLC.row(j));
      //     l_t_list.emplace_back(CLT(j));
      //   }
      // }
      // l_color_list.emplace_back(CLCn.row(CLCn.rows()-1));
      // Eigen::Matrix3d CLC(l_color_list.size(),3);
      // for(int j=0; j<l_color_list.size(); j++){
      //   CLC.row(j)=l_color_list[j];
      // }
      // Eigen::VectorXd CLT(l_t_list.size());
      // for(int j=0; j<l_t_list.size(); j++){
      //   CLT(j) = l_t_list[j];
      // }

    }
  }

}

void infinite::discretize_diffcurv_bezier(const Mat42_list &CPs,
                                  const MatX3d_list &CLCs,
                                  const VecXd_list &CLTs,
                                  const MatX3d_list &CRCs,
                                  const VecXd_list &CRTs,
                                  const Eigen::VectorXi &nel,
                                  const VecXd_list &xel,
                                  const VecXd_list &xecl,
                                  Eigen::MatrixX2d &P,
                                  Eigen::MatrixX2d &C,
                                  Eigen::MatrixX2d &N,
                                  Eigen::MatrixX2i &E,
                                  Eigen::VectorXd &L,
                                  Eigen::MatrixX3d &u_l,
                                  Eigen::MatrixX3d &u_r)
{
  infinite::line_segments_bezier(CPs,nel,xel,xecl,P,C,N,E,L);

  infinite::diffcurv_color(CLCs,CLTs,xecl,u_l);
  infinite::diffcurv_color(CRCs,CRTs,xecl,u_r);
}


void infinite::discretize_diffcurv(const Mat42_list &CPs,
                                  const MatX3d_list &CLCs,
                                  const VecXd_list &CLTs,
                                  const MatX3d_list &CRCs,
                                  const VecXd_list &CRTs,
                                  const Eigen::VectorXi &nel,
                                  const VecXd_list &xel,
                                  const VecXd_list &xecl,
                                  Eigen::MatrixX2d &P,
                                  Eigen::MatrixX2d &C,
                                  Eigen::MatrixX2d &N,
                                  Eigen::MatrixX2i &E,
                                  Eigen::VectorXd &L,
                                  Eigen::MatrixX3d &u_l,
                                  Eigen::MatrixX3d &u_r)
{
  infinite::line_segments_arclen(CPs,nel,xel,xecl,P,C,N,E,L);

  infinite::diffcurv_color(CLCs,CLTs,xecl,u_l);
  infinite::diffcurv_color(CRCs,CRTs,xecl,u_r);
}

void infinite::discretize_diffcurv(const Mat42_list &CPs,
                                  const MatX3d_list &CLCs,
                                  const VecXd_list &CLTs,
                                  const MatX3d_list &CRCs,
                                  const VecXd_list &CRTs,
                                  const Eigen::VectorXd &xe,
                                  const Eigen::VectorXd &xec,
                                  Eigen::MatrixX2d &P,
                                  Eigen::MatrixX2d &C,
                                  Eigen::MatrixX2d &N,
                                  Eigen::MatrixX2i &E,
                                  Eigen::VectorXd &L,
                                  Eigen::MatrixX3d &u_l,
                                  Eigen::MatrixX3d &u_r)
{
  infinite::line_segments_arclen(CPs,xe,xec,P,C,N,E,L);

  infinite::diffcurv_color(CLCs,CLTs,xec,u_l);
  infinite::diffcurv_color(CRCs,CRTs,xec,u_r);
}


void infinite::discretize_diffcurv(const Mat42_list &CPs,
                                  const MatX3d_list &CLCs,
                                  const VecXd_list &CLTs,
                                  const MatX3d_list &CRCs,
                                  const VecXd_list &CRTs,
                                  const Eigen::VectorXd &xe,
                                  const Eigen::VectorXd &xec,
                                  const Eigen::VectorXd &xg,
                                  Eigen::MatrixX2d &P,
                                  Eigen::MatrixX2d &C,
                                  Eigen::MatrixX2d &Q,
                                  Eigen::MatrixX2d &N,
                                  Eigen::MatrixX2i &E,
                                  Eigen::VectorXd &L,
                                  Eigen::MatrixX3d &u_l,
                                  Eigen::MatrixX3d &u_r)
{
  infinite::line_segments_arclen(CPs,xe,xec,P,C,N,E,L);
  infinite::bezier_curves_points(CPs,xg,Q);

  infinite::diffcurv_color(CLCs,CLTs,xg,u_l);
  infinite::diffcurv_color(CRCs,CRTs,xg,u_r);
}

void infinite::rediscretize_diffcurv(const Eigen::VectorXd &xs,
                                    const Eigen::VectorXd &xsc,
                                    const Eigen::VectorXd &xg,
                                    const Eigen::VectorXi &subI,
                                    const Eigen::VectorXi &origI,
                                    const double &sub_t,
                                    Mat42_list &CPs,
                                    MatX3d_list &CLCs,
                                    VecXd_list &CLTs,
                                    MatX3d_list &CRCs,
                                    VecXd_list &CRTs,
                                    Eigen::MatrixX2d &P,
                                    Eigen::MatrixX2d &C,
                                    Eigen::MatrixX2d &Q,
                                    Eigen::MatrixX2d &N,
                                    Eigen::MatrixX2i &E,
                                    Eigen::VectorXd &L,
                                    Eigen::MatrixX3d &u_l,
                                    Eigen::MatrixX3d &u_r)
{
  int ns = xsc.size();
  int ng = xg.size();

  Mat42_list sCPs(subI.size());
  MatX3d_list sCLCs(subI.size());
  VecXd_list sCLTs(subI.size());
  MatX3d_list sCRCs(subI.size());
  VecXd_list sCRTs(subI.size());

  for(int i=0; i<subI.size(); i++){
    sCPs[i]=CPs[subI(i)];
    sCLCs[i]=CLCs[subI(i)];
    sCLTs[i]=CLTs[subI(i)];
    sCRCs[i]=CRCs[subI(i)];
    sCRTs[i]=CRTs[subI(i)];
  }
  infinite::subdivide_diffcurv(sCPs,sCLCs,sCLTs,sCRCs,sCRTs,sub_t);

  Eigen::MatrixX2d sP,sN,sC,sQ;
  Eigen::MatrixX2i sE;
  Eigen::VectorXd sL;

  Eigen::MatrixX3d us_l, us_r;
  infinite::line_segments_arclen(sCPs,xs,xsc,sP,sC,sN,sE,sL);
  infinite::bezier_curves_points(sCPs,xg,sQ);
  infinite::diffcurv_color(sCLCs,sCLTs,xg,us_l);
  infinite::diffcurv_color(sCRCs,sCRTs,xg,us_r);

  Eigen::VectorXi ois_s(origI.size()*ns);
  Eigen::VectorXi ois_p(origI.size()*(ns+1));
  Eigen::VectorXi ois_g(origI.size()*ng);

  for(int i=0; i<origI.size(); i++){
    for(int j=0; j<ns; j++){
      ois_s(ns*i+j) = ns*origI(i)+j;
      ois_p((ns+1)*i+j) = (ns+1)*origI(i)+j;
    }
    ois_p((ns+1)*i+ns) = (ns+1)*origI(i)+ns;
    for(int j=0; j<ng; j++){
      ois_g(ng*i+j) = ng*origI(i)+j;
    }
  }
  Eigen::MatrixX2d oP,oN,oC,oQ;
  Eigen::VectorXd oL;
  Eigen::MatrixX3d uo_l, uo_r;

  igl::slice(P,ois_p,1,oP);
  igl::slice(C,ois_s,1,oC);
  igl::slice(N,ois_s,1,oN);
  igl::slice(L,ois_s,oL);
  igl::slice(Q,ois_g,1,oQ);
  igl::slice(u_l,ois_g,1,uo_l);
  igl::slice(u_r,ois_g,1,uo_r);

  P.resize(oP.rows()+sP.rows(),2);
  C.resize(oC.rows()+sC.rows(),2);
  N.resize(oN.rows()+sN.rows(),2);
  L.resize(oL.size()+sL.size());
  Q.resize(oQ.rows()+sQ.rows(),2);
  u_l.resize(uo_l.rows()+us_l.rows(),3);
  u_r.resize(uo_r.rows()+us_r.rows(),3);

  P<<oP,sP;
  C<<oC,sC;
  N<<oN,sN;
  L<<oL,sL;
  Q<<oQ,sQ;
  u_l<<uo_l,us_l;
  u_r<<uo_r,us_r;

  E = edge_indices(2*subI.size()+origI.size(),ns);

  Mat42_list nCPs(2*subI.size()+origI.size());
  MatX3d_list nCLCs(2*subI.size()+origI.size());
  VecXd_list nCLTs(2*subI.size()+origI.size());
  MatX3d_list nCRCs(2*subI.size()+origI.size());
  VecXd_list nCRTs(2*subI.size()+origI.size());

  for(int i=0; i<origI.size(); i++){
    nCPs[i] = CPs[origI(i)];
    nCLCs[i] = CLCs[origI(i)];
    nCLTs[i] = CLTs[origI(i)];
    nCRCs[i] = CRCs[origI(i)];
    nCRTs[i] = CRTs[origI(i)];
  }
  for(int i=0; i<sCPs.size(); i++){
    int ni = origI.size()+i;
    nCPs[ni] = sCPs[i];
    nCLCs[ni] = sCLCs[i];
    nCLTs[ni] = sCLTs[i];
    nCRCs[ni] = sCRCs[i];
    nCRTs[ni] = sCRTs[i];
  }
  CPs=nCPs; CLCs=nCLCs; CLTs=nCLTs; CRCs=nCRCs; CRTs=nCRTs;

}