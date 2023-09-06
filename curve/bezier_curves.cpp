#include "bezier_curves.h"


Eigen::RowVector2d infinite::point_bezier(const Mat42 &CP, const double& t)
{
  double t0 = std::pow((1.0-t),3);
  double t1 = 3.0*std::pow((1-t),2)*t;
  double t2 = 3.0*(1-t)*std::pow(t,2);
  double t3 = std::pow(t,3);

  Eigen::RowVector2d P;
  P(0) = t0*CP(0,0)+t1*CP(1,0)+t2*CP(2,0)+t3*CP(3,0);
  P(1) = t0*CP(0,1)+t1*CP(1,1)+t2*CP(2,1)+t3*CP(3,1);
  return P;
}

Eigen::RowVector2d infinite::normal_bezier(const Mat42 &CP, const double& t)
{
  double t1 = 3*std::pow((1-t),2);
  double t2 = 6*(1-t)*t;
  double t3 = 3*std::pow(t,2);

  Eigen::RowVector2d dP;
  dP(0) = (CP(1,1)-CP(0,1))*t1+(CP(2,1)-CP(1,1))*t2+(CP(3,1)-CP(2,1))*t3;
  dP(1) = -((CP(1,0)-CP(0,0))*t1+(CP(2,0)-CP(1,0))*t2+(CP(3,0)-CP(2,0))*t3);

  return dP;
}

Eigen::RowVector2d infinite::derivative_bezier(const Mat42 &CP, const double& t)
{
  double t1 = 3*std::pow((1-t),2);
  double t2 = 6*(1-t)*t;
  double t3 = 3*std::pow(t,2);

  Eigen::RowVector2d dP;
  dP(0) = (CP(1,0)-CP(0,0))*t1+(CP(2,0)-CP(1,0))*t2+(CP(3,0)-CP(2,0))*t3;
  dP(1) = (CP(1,1)-CP(0,1))*t1+(CP(2,1)-CP(1,1))*t2+(CP(3,1)-CP(2,1))*t3;

  return dP;
}

Eigen::MatrixX2d infinite::points_bezier(const Mat42 &CP, const Eigen::VectorXd& T)
{
  Eigen::VectorXd T0 = (1.0-T.array()).pow(3);
  Eigen::VectorXd T1 = 3.0*(1.0-T.array()).pow(2)*T.array();
  Eigen::VectorXd T2 = 3.0*(1.0-T.array())*T.array().pow(2);
  Eigen::VectorXd T3 = T.array().pow(3);

  Eigen::MatrixX2d P(T.size(),2);
  P.col(0) = T0*CP(0,0)+T1*CP(1,0)+T2*CP(2,0)+T3*CP(3,0);
  P.col(1) = T0*CP(0,1)+T1*CP(1,1)+T2*CP(2,1)+T3*CP(3,1);
  return P;
}

Eigen::MatrixX2d infinite::normal_bezier(const Mat42 &CP, const Eigen::VectorXd& T)
{
  Eigen::VectorXd T1 = 3*(1-T.array()).square();
  Eigen::VectorXd T2 = 6*(1-T.array())*T.array();
  Eigen::VectorXd T3 = 3*T.array().square();

  Eigen::MatrixX2d dP(T.size(),2);
  dP.col(0) = (CP(1,1)-CP(0,1))*T1+(CP(2,1)-CP(1,1))*T2+(CP(3,1)-CP(2,1))*T3;
  dP.col(1) = -((CP(1,0)-CP(0,0))*T1+(CP(2,0)-CP(1,0))*T2+(CP(3,0)-CP(2,0))*T3);

  return dP;
}

Eigen::MatrixX2d infinite::derivative_bezier(const Mat42 &CP, const Eigen::VectorXd& T)
{
  Eigen::VectorXd T1 = 3*(1-T.array()).square();
  Eigen::VectorXd T2 = 6*(1-T.array())*T.array();
  Eigen::VectorXd T3 = 3*T.array().square();

  Eigen::MatrixX2d dP(T.size(),2);
  dP.col(0) = (CP(1,0)-CP(0,0))*T1+(CP(2,0)-CP(1,0))*T2+(CP(3,0)-CP(2,0))*T3;
  dP.col(1) = (CP(1,1)-CP(0,1))*T1+(CP(2,1)-CP(1,1))*T2+(CP(3,1)-CP(2,1))*T3;

  return dP;
}

Eigen::MatrixX2d infinite::second_derivative_bezier(const Mat42 &CP, const Eigen::VectorXd& T)
{
  Eigen::VectorXd T1 = 6*(1-T.array());
  Eigen::VectorXd T2 = 6*T.array();

  Eigen::MatrixX2d ddP(T.size(),2);
  ddP.col(0) = T1*(CP(2,0)-2*CP(1,0)+CP(0,0))+T2*(CP(3,0)-2*CP(2,0)+CP(1,0));
  ddP.col(1) = T1*(CP(2,1)-2*CP(1,1)+CP(0,1))+T2*(CP(3,1)-2*CP(2,0)+CP(1,1));

  return ddP;
}


void infinite::line_segments_bezier(const Mat42_list &CPs,
                        const Eigen::VectorXi &nel,
                        const VecXd_list &xel,
                        const VecXd_list &xecl,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L)
{
  int nb = CPs.size();
  int np=0;
  Eigen::VectorXi sil(nel.size());
  for(int k=0; k<nel.size(); k++){
    sil(k)=np;
    np=np+nel(k);
  }

  P.resize(np+nb,2);
  C.resize(np,2);
  N.resize(np,2);
  L.resize(np);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){
    int ne = nel(k);
    int si = sil(k);
    Eigen::VectorXd xe = xel[k];
    Eigen::VectorXd xec = xecl[k];

    Eigen::MatrixX2d Pk = points_bezier(CPs[k],xe);
    Eigen::MatrixX2d Ck = points_bezier(CPs[k],xec);
    Eigen::MatrixX2d Nk = normal_bezier(CPs[k],xec);
    Eigen::VectorXd Lk = cubic_length(CPs[k],xe.head(ne),xe.tail(ne));

    P.block(si+k,0,ne+1,2)=Pk;
    C.block(si,0,ne,2)=Ck;
    N.block(si,0,ne,2)=Nk;
    L.segment(si,ne)=Lk;
  }
  N.rowwise().normalize();
  edge_indices(nel,E);
}


void infinite::line_segments_arclen(const Mat42_list &CPs,
                        const Eigen::VectorXi &nel,
                        const VecXd_list &xel,
                        const VecXd_list &xecl,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L)
{
  int nb = CPs.size();
  int np=0;
  Eigen::VectorXi sil(nel.size());
  for(int k=0; k<nel.size(); k++){
    sil(k)=np;
    np=np+nel(k);
  }

  P.resize(np+nb,2);
  C.resize(np,2);
  N.resize(np,2);
  L.resize(np);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){
    int ne = nel(k);
    int si = sil(k);
    Eigen::VectorXd xe = xel[k];
    Eigen::VectorXd xec = xecl[k];

    Eigen::VectorXd xen, xecn;
    bezier_arclen_param(CPs[k],xe,xen);
    bezier_arclen_param(CPs[k],xec,xecn);
    xen(0)=0;
    xen(xen.size()-1)=1;

    Eigen::MatrixX2d Pk = points_bezier(CPs[k],xen);
    Eigen::MatrixX2d Ck = points_bezier(CPs[k],xecn);
    Eigen::MatrixX2d Nk = normal_bezier(CPs[k],xecn);
    Eigen::VectorXd Lk = cubic_length(CPs[k],xen.head(ne),xen.tail(ne));

    P.block(si+k,0,ne+1,2)=Pk;
    C.block(si,0,ne,2)=Ck;
    N.block(si,0,ne,2)=Nk;
    L.segment(si,ne)=Lk;
  }
  N.rowwise().normalize();
  edge_indices(nel,E);
}

void infinite::line_segments_arclen(const Mat42_list &CPs,
                        const Eigen::VectorXi &nel,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L)
{
  int nb = nel.size();
  int np=0;
  Eigen::VectorXi sil(nel.size());
  for(int k=0; k<nel.size(); k++){
    sil(k)=np;
    np=np+nel(k);
  }

  P.resize(np+nb,2);
  C.resize(np,2);
  N.resize(np,2);
  L.resize(np);

  #pragma omp parallel for
  for(int k=0; k<nel.size(); k++){
    int ne = nel(k);
    int si = sil(k);
    Eigen::VectorXd xe = Eigen::VectorXd::LinSpaced(ne+1,0,1);
    Eigen::VectorXd xec = (xe.head(ne)+xe.tail(ne))/2;

    Eigen::VectorXd xen, xecn;
    bezier_arclen_param(CPs[k],xe,xen);
    bezier_arclen_param(CPs[k],xec,xecn);

    Eigen::MatrixX2d Pk = points_bezier(CPs[k],xen);
    Eigen::MatrixX2d Ck = points_bezier(CPs[k],xecn);
    Eigen::MatrixX2d Nk = normal_bezier(CPs[k],xecn);
    Eigen::VectorXd Lk = cubic_length(CPs[k],xen.head(ne),xen.tail(ne));

    P.block(si+k,0,ne+1,2)=Pk;
    C.block(si,0,ne,2)=Ck;
    N.block(si,0,ne,2)=Nk;
    L.segment(si,ne)=Lk;
  }
  N.rowwise().normalize();
  edge_indices(nel,E);
}

void infinite::line_segments_arclen(const Mat42_list &CPs,
                        const Eigen::VectorXd &xe,
                        const Eigen::VectorXd &xec,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L)
{
  int nb = CPs.size();
  int ne = xec.size();

  P.resize(nb*(ne+1),2);
  C.resize(nb*ne,2);
  N.resize(nb*ne,2);
  L.resize(nb*ne);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){

    Eigen::VectorXd xen, xecn;
    bezier_arclen_param(CPs[k],xe,xen);
    bezier_arclen_param(CPs[k],xec,xecn);

    Eigen::MatrixX2d Pk = points_bezier(CPs[k],xen);
    Eigen::MatrixX2d Ck = points_bezier(CPs[k],xecn);
    Eigen::MatrixX2d Nk = normal_bezier(CPs[k],xecn);
    Eigen::VectorXd Lk = cubic_length(CPs[k],xen.head(ne),xen.tail(ne));

    P.block(k*(ne+1),0,ne+1,2)=Pk;
    C.block(k*ne,0,ne,2)=Ck;
    N.block(k*ne,0,ne,2)=Nk;
    L.segment(k*ne,ne)=Lk;
  }
  N.rowwise().normalize();
  E = edge_indices(nb,ne);
}

void infinite::line_segments_arclen(const Mat42_list &CPs,
                        const int &ne,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L)
{
  int nb = CPs.size();

  Eigen::VectorXd xe = Eigen::VectorXd::LinSpaced(ne+1,0,1);
  Eigen::VectorXd xec = (xe.head(ne)+xe.tail(ne))/2;

  P.resize(nb*(ne+1),2);
  C.resize(nb*ne,2);
  N.resize(nb*ne,2);
  L.resize(nb*ne);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){

    Eigen::VectorXd xen, xecn;
    bezier_arclen_param(CPs[k],xe,xen);
    bezier_arclen_param(CPs[k],xec,xecn);

    Eigen::MatrixX2d Pk = points_bezier(CPs[k],xen);
    Eigen::MatrixX2d Ck = points_bezier(CPs[k],xecn);
    Eigen::MatrixX2d Nk = normal_bezier(CPs[k],xecn);
    Eigen::VectorXd Lk = cubic_length(CPs[k],xe.head(ne),xe.tail(ne));

    P.block(k*(ne+1),0,ne+1,2)=Pk;
    C.block(k*ne,0,ne,2)=Ck;
    N.block(k*ne,0,ne,2)=Nk;
    L.segment(k*ne,ne)=Lk;
  }
  N.rowwise().normalize();
  E = edge_indices(nb,ne);
}


void infinite::bezier_curves(const Mat42_list &CPs, 
                   const Eigen::VectorXd &T,
                   Eigen::MatrixX2d &P,
                   Eigen::MatrixX2d &N,
                   Eigen::VectorXd &CR)
{
  int nb = CPs.size();
  int np = T.size();
  P.resize(nb*np,2);
  N.resize(nb*np,2);
  CR.resize(nb*np);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){
    Eigen::VectorXd TN;
    bezier_arclen_param(CPs[k],T,TN);
    Eigen::MatrixX2d Pk = points_bezier(CPs[k],TN);
    Eigen::MatrixX2d dPk = derivative_bezier(CPs[k],TN);
    Eigen::MatrixX2d ddPk = second_derivative_bezier(CPs[k],TN);
    Eigen::MatrixX2d Nk(dPk.rows(),2);
    Nk.col(0) = dPk.col(1);
    Nk.col(1) = -dPk.col(0);
    Eigen::VectorXd cur;
    cur = (dPk.col(0).array()*ddPk.col(1).array()-dPk.col(1).array()*ddPk.col(0).array())/
          (dPk.col(0).array().pow(2)+dPk.col(1).array().pow(2)).pow(1.5).abs();
    P.block(np*k,0,np,2)=Pk;
    N.block(np*k,0,np,2)=Nk;
    CR.segment(np*k,np) =cur;
  }
}



void infinite::bezier_curves_points(const Mat42_list &CPs, 
                          const Eigen::VectorXd &T,
                          const curve_param param,
                          Eigen::MatrixX2d &P)
{
  int nb = CPs.size();
  int np = T.size();
  P.resize(nb*np,2);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){
    Eigen::VectorXd TN = T;
    if(param==arclen){
      bezier_arclen_param(CPs[k],T,TN);
    }
    Eigen::MatrixX2d Pk = points_bezier(CPs[k],TN);
    P.block(np*k,0,np,2)=Pk;
  }
}

void infinite::bezier_curves_normals(const Mat42_list &CPs, 
                            const Eigen::VectorXd &T,
                            const curve_param param,
                            Eigen::MatrixX2d &N)
{
  int nb = CPs.size();
  int np = T.size();
  N.resize(nb*np,2);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){
    Eigen::VectorXd TN = T;
    if(param==arclen){
      bezier_arclen_param(CPs[k],T,TN);
    }
    Eigen::MatrixX2d dPk = derivative_bezier(CPs[k],TN);
    Eigen::MatrixX2d Nk(dPk.rows(),2);
    Nk.col(0) = dPk.col(1);
    Nk.col(1) = -dPk.col(0);
    N.block(np*k,0,np,2)=Nk;
  }
}


void infinite::bezier_curves_points(const Mat42_list &CPs, 
                          const Eigen::VectorXd &T,
                          Eigen::MatrixX2d &P)
{
  curve_param param = arclen;
  bezier_curves_points(CPs,T,param,P);
}

void infinite::bezier_curves_normals(const Mat42_list &CPs, 
                          const Eigen::VectorXd &T,
                          Eigen::MatrixX2d &N)
{
  curve_param param = arclen;
  bezier_curves_normals(CPs,T,param,N);
}


////////////////////////////////////////////////////////



void infinite::bezier_curves_points(const Mat42_list &CPs, 
                          const Eigen::VectorXd &T,
                          const curve_param param,
                          MatX2d &P)
{
  int nb = CPs.size();
  int np = T.size();
  P.resize(nb*np,2);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){
    Eigen::VectorXd TN = T;
    if(param==arclen){
      bezier_arclen_param(CPs[k],T,TN);
    }
    Eigen::MatrixX2d Pk = points_bezier(CPs[k],TN);
    P.block(np*k,0,np,2)=Pk;
  }
}

void infinite::bezier_curves_normals(const Mat42_list &CPs, 
                            const Eigen::VectorXd &T,
                            const curve_param param,
                            MatX2d &N)
{
  int nb = CPs.size();
  int np = T.size();
  N.resize(nb*np,2);

  #pragma omp parallel for
  for(int k=0; k<nb; k++){
    Eigen::VectorXd TN = T;
    if(param==arclen){
      bezier_arclen_param(CPs[k],T,TN);
    }
    Eigen::MatrixX2d dPk = derivative_bezier(CPs[k],TN);
    Eigen::MatrixX2d Nk(dPk.rows(),2);
    Nk.col(0) = dPk.col(1);
    Nk.col(1) = -dPk.col(0);
    N.block(np*k,0,np,2)=Nk;
  }
}


void infinite::bezier_curves_points(const Mat42_list &CPs, 
                          const Eigen::VectorXd &T,
                          MatX2d &P)
{
  curve_param param = arclen;
  bezier_curves_points(CPs,T,param,P);
}

void infinite::bezier_curves_normals(const Mat42_list &CPs, 
                          const Eigen::VectorXd &T,
                          MatX2d &N)
{
  curve_param param = arclen;
  bezier_curves_normals(CPs,T,param,N);
}