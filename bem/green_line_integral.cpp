#include "green_line_integral.h"


double infinite::green_line_integral(const Eigen::MatrixX2d &P,
                         const Eigen::MatrixX2i &E,
                         const int &id,
                         const double  &l,
                         const Eigen::RowVector2d &Q)
{
  Eigen::RowVector2d a = P.row(E(id,1))-P.row(E(id,0));;
  double q = l*l;
  Eigen::RowVector2d b = P.row(E(id,0))-Q;

  double s = b.squaredNorm();
  double r = 2*a.dot(b);
  r = r*(l/a.norm());
  double srt = sqrt(4*s*q-r*r);
  if(srt==0)
    srt = 1e-30;
  double l0 = log(s);
  double l1 = log(s+q+r);

  double a0 = atan2(r,srt)/srt;
  double a1 = atan2(2*q+r,srt)/srt;

  double g;
  if(std::isnan(a0)){
    g=0; 
  }
  else if(std::isnan(a1)){
    g=0; 
  }
  else{
    double a10 = a1-a0;
    double l10 = l1-l0;

    g=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/2;//(4*igl::PI);
  }
  return g;
}




double infinite::green_line_integral(const Eigen::Matrix2d &PEs,
                         const Eigen::RowVector2d &Q)
{
  Eigen::RowVector2d a = PEs.row(1)-PEs.row(0);
  double l = a.norm();
  double q = l*l;
  Eigen::RowVector2d b = PEs.row(0)-Q;

  double s = b.squaredNorm();
  double r = 2*a.dot(b);
  double srt = sqrt(4*s*q-r*r);
  if(srt==0)
    srt = 1e-30;
  double l0 = log(s);
  double l1 = log(s+q+r);

  double a0 = atan2(r,srt)/srt;
  double a1 = atan2(2*q+r,srt)/srt;

  double g;
  if(std::isnan(a0)){
    g=0; 
  }
  else if(std::isnan(a1)){
    g=0; 
  }
  else{
    double a10 = a1-a0;
    double l10 = l1-l0;
    g=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/2;//(4*igl::PI);
  }
  return g;
}



double infinite::green_line_integral(const Eigen::Matrix2d &PEs,
                         const double  &l,
                         const Eigen::RowVector2d &Q)
{
  Eigen::RowVector2d a = PEs.row(1)-PEs.row(0);
  double q = l*l;
  Eigen::RowVector2d b = PEs.row(0)-Q;

  double s = b.squaredNorm();
  double r = 2*a.dot(b);
  r = r*(l/a.norm());
  double srt = sqrt(4*s*q-r*r);
  if(srt==0)
    srt = 1e-30;
  double l0 = log(s);
  double l1 = log(s+q+r);

  double a0 = atan2(r,srt)/srt;
  double a1 = atan2(2*q+r,srt)/srt;

  double g;
  if(std::isnan(a0)){
    g=0; 
  }
  else if(std::isnan(a1)){
    g=0; 
  }
  else{
    double a10 = a1-a0;
    double l10 = l1-l0;

    g=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/2;//(4*igl::PI);
  }
  return g;
}





void infinite::green_line_integral(const Mat2d_list &PEs,
                         const Eigen::VectorXd &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G)
{
  G.resize(PEs.size());

  for(int j=0; j<PEs.size(); j++){
    Eigen::RowVector2d a = PEs[j].row(1)-PEs[j].row(0);

    double l = Ls(j);
    double q = l*l;
    Eigen::RowVector2d b = PEs[j].row(0)-Q;

    double s = b.squaredNorm();
    double r = 2*a.dot(b);
    r = r*(l/a.norm());
    double srt = sqrt(4*s*q-r*r);
    if(srt==0)
      srt = 1e-30;
    double l0 = log(s);
    double l1 = log(s+q+r);

    double a0 = atan2(r,srt)/srt;
    double a1 = atan2(2*q+r,srt)/srt;

    if(std::isnan(a0)){
      G(j)=0; 
    }
    else if(std::isnan(a1)){
      G(j)=0; 
    }
    else{
      double a10 = a1-a0;
      double l10 = l1-l0;
      G(j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/(4*igl::PI);
    }
  }
}



void infinite::green_line_integral(const Mat2d_list &PEs,
                         const Eigen::MatrixX2d &N,
                         const Eigen::VectorXd &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G,
                         Eigen::RowVectorXd &F)
{
  G.resize(PEs.size());
  F.resize(PEs.size());

  for(int j=0; j<PEs.size(); j++){
    Eigen::RowVector2d a = PEs[j].row(1)-PEs[j].row(0);
    // int id = ids[j];
    // Eigen::RowVector2d a = P.row(E(id,1))-P.row(E(id,0));
    // double q = a.squaredNorm();
    double l = Ls(j);
    double q = l*l;
    Eigen::RowVector2d b = PEs[j].row(0)-Q;

    double s = b.squaredNorm();
    double r = 2*a.dot(b);
    r = r*(l/a.norm());
    double srt = sqrt(4*s*q-r*r);
    if(srt==0)
      srt = 1e-30;
    double l0 = log(s);
    double l1 = log(s+q+r);

    double a0 = atan2(r,srt)/srt;
    double a1 = atan2(2*q+r,srt)/srt;

    if(std::isnan(a0)){
      G(j)=0; F(j)=0;
    }
    else if(std::isnan(a1)){
      G(j)=0; F(j)=0;
    }
    else{
      double a10 = a1-a0;
      double l10 = l1-l0;

      Eigen::RowVector2d n = N.row(j);
      double bn = b.dot(n);

      G(j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/(4*igl::PI);
      F(j)=l*(bn*a10)/igl::PI;
    }
  }
}

void infinite::green_line_integral_test(const Mat2d_list &PEs,
                         const Eigen::MatrixX2d &N,
                         const std::vector<double>  &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G,
                         Eigen::RowVectorXd &Gf,
                         Eigen::RowVectorXd &F)
{
  G.resize(PEs.size());
  Gf.resize(PEs.size());
  F.resize(PEs.size());

  for(int j=0; j<PEs.size(); j++){
    Eigen::RowVector2d a = PEs[j].row(1)-PEs[j].row(0);
    // int id = ids[j];
    // Eigen::RowVector2d a = P.row(E(id,1))-P.row(E(id,0));
    // double q = a.squaredNorm();
    double l = Ls[j];
    double q = l*l;
    Eigen::RowVector2d b = PEs[j].row(0)-Q;

    double s = b.squaredNorm();
    double r = 2*a.dot(b);
    r = r*(l/a.norm());
    double srt = sqrt(4*s*q-r*r);
    if(srt==0)
      srt = 1e-30;
    double l0 = log(s);
    double l1 = log(s+q+r);

    double a0 = atan2(r,srt)/srt;
    double a1 = atan2(2*q+r,srt)/srt;

    if(std::isnan(a0)){
      G(j)=0; F(j)=0;
    }
    else if(std::isnan(a1)){
      G(j)=0; F(j)=0;
    }
    else{
      double a10 = a1-a0;
      double l10 = l1-l0;

      Eigen::RowVector2d n = N.row(j);
      double bn = b.dot(n);

      G(j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/2;//(4*igl::PI);
      F(j)=l*(bn*a10)*2;
      Gf(j)=(log(2*l*sqrt(s+q+r)+2*q+r)-log(2*l*sqrt(s)+r))/l;
    }
  }
}


void infinite::green_line_integral(const Mat2d_list &PEs,
                         const Eigen::MatrixX2d &N,
                         const std::vector<double>  &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G,
                         Eigen::RowVectorXd &F)
{
  G.resize(PEs.size());
  F.resize(PEs.size());

  for(int j=0; j<PEs.size(); j++){
    Eigen::RowVector2d a = PEs[j].row(1)-PEs[j].row(0);
    // int id = ids[j];
    // Eigen::RowVector2d a = P.row(E(id,1))-P.row(E(id,0));
    // double q = a.squaredNorm();
    double l = Ls[j];
    double q = l*l;
    Eigen::RowVector2d b = PEs[j].row(0)-Q;

    double s = b.squaredNorm();
    double r = 2*a.dot(b);
    r = r*(l/a.norm());
    double srt = sqrt(4*s*q-r*r);
    if(srt==0)
      srt = 1e-30;
    double l0 = log(s);
    double l1 = log(s+q+r);

    double a0 = atan2(r,srt)/srt;
    double a1 = atan2(2*q+r,srt)/srt;

    if(std::isnan(a0)){
      G(j)=0; F(j)=0;
    }
    else if(std::isnan(a1)){
      G(j)=0; F(j)=0;
    }
    else{
      double a10 = a1-a0;
      double l10 = l1-l0;

      Eigen::RowVector2d n = N.row(j);
      double bn = b.dot(n);

      G(j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/(4*igl::PI);
      F(j)=l*(bn*a10)/igl::PI;
    }
  }
}

void infinite::green_line_integral(const Mat2d_list &PEs,
                         const std::vector<double>  &Ls,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G)
{
  G.resize(PEs.size());

  for(int j=0; j<PEs.size(); j++){
    Eigen::RowVector2d a = PEs[j].row(1)-PEs[j].row(0);
    // int id = ids[j];
    // Eigen::RowVector2d a = P.row(E(id,1))-P.row(E(id,0));
    // double q = a.squaredNorm();
    double l = Ls[j];
    double q = l*l;
    Eigen::RowVector2d b = PEs[j].row(0)-Q;

    double s = b.squaredNorm();
    double r = 2*a.dot(b);
    r = r*(l/a.norm());
    double srt = sqrt(4*s*q-r*r);
    if(srt==0)
      srt = 1e-30;
    double l0 = log(s);
    double l1 = log(s+q+r);

    double a0 = atan2(r,srt)/srt;
    double a1 = atan2(2*q+r,srt)/srt;

    if(std::isnan(a0)){
      G(j)=0; 
    }
    else if(std::isnan(a1)){
      G(j)=0; 
    }
    else{
      double a10 = a1-a0;
      double l10 = l1-l0;

      G(j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/2;//(4*igl::PI);
    }
  }
}




void infinite::green_line_integral(const MatX2d &P,
                         const MatX2i &E,
                         const MatX2d &N,
                         const Eigen::VectorXd  &L,
                         const std::vector<int> &ids,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G,
                         Eigen::RowVectorXd &F)
{
  G.resize(ids.size());
  F.resize(ids.size());

  for(int j=0; j<ids.size(); j++){
    int id = ids[j];
    Eigen::RowVector2d a = P.row(E(id,1))-P.row(E(id,0));
    Eigen::RowVector2d n = N.row(id);
    //double q = a.squaredNorm();
    double l = L(id);
    double q=l*l;

    Eigen::RowVector2d b = P.row(E(id,0))-Q;
    double s = b.squaredNorm();
    double r = 2*a.dot(b);
    r = r*(l/a.norm());
    // double r = l*l;
    double srt = sqrt(4*s*q-r*r);
    if(srt==0)
      srt = 1e-30;
    double l0 = log(s);
    double l1 = log(s+q+r);

    double a0 = atan2(r,srt)/srt;
    double a1 = atan2(2*q+r,srt)/srt;



    if(std::isnan(a0)){
      G(j)=0; F(j)=0;
    }
    else if(std::isnan(a1)){
      G(j)=0; F(j)=0;
    }
    else{
      double a10 = a1-a0;
      double l10 = l1-l0;

      double bn = b.dot(n);

      G(j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/(4*igl::PI);
      F(j)=l*(bn*a10)/igl::PI;
    }
  }
}


void infinite::green_line_integral(const Eigen::MatrixX2d &P,
                         const Eigen::MatrixX2i &E,
                         const Eigen::MatrixX2d &N,
                         const Eigen::VectorXd  &L,
                         const std::vector<int> &ids,
                         const Eigen::RowVector2d &Q,
                         Eigen::RowVectorXd &G,
                         Eigen::RowVectorXd &F)
{
  G.resize(ids.size());
  F.resize(ids.size());

  for(int j=0; j<ids.size(); j++){
    int id = ids[j];
    Eigen::RowVector2d a = P.row(E(id,1))-P.row(E(id,0));
    Eigen::RowVector2d n = N.row(id);
    // double q = a.squaredNorm();
    double l = L(id);
    double q = l*l;

    Eigen::RowVector2d b = P.row(E(id,0))-Q;
    double s = b.squaredNorm();
    double r = 2*a.dot(b);
    r = r*(l/a.norm());
    double srt = sqrt(4*s*q-r*r);
    if(srt==0)
      srt = 1e-30;
    double l0 = log(s);
    double l1 = log(s+q+r);

    double a0 = atan2(r,srt)/srt;
    double a1 = atan2(2*q+r,srt)/srt;



    if(std::isnan(a0)){
      G(j)=0; F(j)=0;
    }
    else if(std::isnan(a1)){
      G(j)=0; F(j)=0;
    }
    else{
      double a10 = a1-a0;
      double l10 = l1-l0;

      double bn = b.dot(n);

      G(j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/(4*igl::PI);
      F(j)=l*(bn*a10)/igl::PI;
    }
  }
}

void infinite::green_line_integral(const Eigen::MatrixX2d &P,
                         const Eigen::MatrixX2i &E,
                         const Eigen::MatrixX2d &N,
                         const Eigen::VectorXd  &L,
                         const Eigen::MatrixX2d &Q,
                         Eigen::MatrixXd &G,
                         Eigen::MatrixXd &F)
{
  G.resize(Q.rows(),E.rows());
  F.resize(Q.rows(),E.rows());

  for(int j=0; j<E.rows(); j++){
    Eigen::RowVector2d a = P.row(E(j,1))-P.row(E(j,0));
    Eigen::RowVector2d n = N.row(j);
    // double q = a.squaredNorm();
    double l = L(j);
    double q = l*l;

    for(int i=0; i<Q.rows(); i++){
      Eigen::RowVector2d b = P.row(E(j,0))-Q.row(i);
      double s = b.squaredNorm();
      double r = 2*a.dot(b);
      r = r*(l/a.norm());

      double srt = std::max(sqrt(4*s*q-r*r),1e-20);
      // double srt = sqrt(4*s*q-r*r);

      double l0 = log(s);
      double l1 = log(s+q+r);

      double a0 = atan2(r,srt)/srt;
      double a1 = atan2(2*q+r,srt)/srt;

      if(std::isnan(a0)){
        G(i,j)=0;F(i,j)=0; 
      }
      if(std::isnan(a1)){
        G(i,j)=0;F(i,j)=0; 
      }
      else{
        double a10 = a1-a0;
        double l10 = l1-l0;
        double bn = b.dot(n);
        G(i,j)=-l*((4*s-(r*r)/q)*a10 + (r/(2*q))*l10+l1-2)/(4*igl::PI);
        F(i,j)=l*(bn*a10)/igl::PI;
      }
    }
  }
}
