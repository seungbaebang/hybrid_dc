#include "diffcurv_color.h"

// void infinite::segment_diffcurv_color(const MatX3d_list &CLCl,
//                                 const VecXd_list &CLTl,
//                                 const MatX3d_list &CRCl,
//                                 const VecXd_list &CRTl,
//                                 MatX3d_list &CLCs,
//                                 VecXd_list &CLTs,
//                                 MatX3d_list &CRCs,
//                                 VecXd_list &CRTs)
// {
//   for(int i=0; i<CLCl.size(); i++){
//     Eigen::MatrixX3d CLC = CLCl[i];
//     Eigen::VectorXd CLT = CLTl[i];

    
//   }
// }

Eigen::RowVector3d infinite::query_color(
                    const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const double &t)
{
  Eigen::RowVector3d CN;
  if(X.size()==0 && C.rows()==1){
    // CN = C.row(0);
    return C.row(0);
  }
  for(int i=0; i<X.size(); i++){
    //if(std::abs(X(i)-t)<1e-4)
    if(X(i)==t)
      return C.row(i);
  }

  Eigen::VectorXd lower_diff = t - X.array();
  Eigen::VectorXd upper_diff = X.array() - t;

  double lower_t=1e+20, upper_t=1e+20;
  int lower_id=-1, upper_id =-1;
  for(int i=0; i<lower_diff.size(); i++){
    if(lower_diff(i)<0)
      continue;
    if(lower_diff(i)<lower_t){
      lower_t = lower_diff(i);
      lower_id = i;
    }
  }
  for(int i=0; i<upper_diff.size(); i++){
    if(upper_diff(i)<0)
      continue;
    if(upper_diff(i)<upper_t){
      upper_t = upper_diff(i);
      upper_id = i;
    }
  }

  if(lower_id<0){
    return C.row(upper_id);
  }
  if(upper_id<0){
    return C.row(lower_id);
  }  

  double lt = (t-X(lower_id))/(X(upper_id)-X(lower_id));
  return (C.row(lower_id)*(1.0-lt)+C.row(upper_id)*lt);
}


Eigen::RowVector3d infinite::eval_diffcurv_color(
                    const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const double &t)
{
  Eigen::RowVector3d CR;
  if(X.size()==0){
    CR = (1-t)*C.row(0)+t*C.row(1);
  }
  else{
    Eigen::VectorXd Xn(X.size()+2);
    Xn.segment(1,X.size())=X;
    Xn(0)=0;Xn(Xn.size()-1)=1;
    for(int j=0; j<Xn.size(); j++){
      if(t>=Xn(j) && t<=Xn(j+1)){
        double tn = (t-Xn(j))/(Xn(j+1)-Xn(j));
        CR=(1-tn)*C.row(j)+tn*C.row(j+1);
      }
    }
  }
  return CR;
}

void infinite::eval_diffcurv_color(
                    const Eigen::MatrixX3d &C,
                    const Eigen::VectorXd &X,
                    const Eigen::VectorXd &T,
                    Eigen::MatrixX3d &CR)
{
  int np = T.size();
  CR.resize(np,3);

  if(X.size()==0){
    for(int i=0; i<np; i++){
      CR.row(i) = (1-T(i))*C.row(0)+T(i)*C.row(1);
    }
  }
  else{
    Eigen::VectorXd Xn(X.size()+2);
    Xn.segment(1,X.size())=X;
    Xn(0)=0;Xn(Xn.size()-1)=1;
    for(int i=0; i<np; i++){
      double t=T(i);
      for(int j=0; j<Xn.size(); j++){
        if(t>=Xn(j) && t<=Xn(j+1)){
          double tn = (t-Xn(j))/(Xn(j+1)-Xn(j));
          CR.row(i)=(1-tn)*C.row(j)+tn*C.row(j+1);
        }
      }
    }
  }
}


void infinite::diffcurv_color(const MatX3d_list &Cs, //pre-defined color on curves
                    const VecXd_list &Xs, //pre-defined parametric value of color on curve
                    const VecXd_list &Ts, //parameter value to evaluate colors
                    Eigen::MatrixX3d &CR) //colors evaluated at Ts
{
  int np=0;
  for(int k=0; k<Ts.size(); k++)
    np=np+Ts[k].size();

  CR.resize(np,3);

  int id=0;
  for(int k=0; k<Cs.size(); k++){
    Eigen::MatrixX3d Ck = Cs[k];
    Eigen::VectorXd Xk = Xs[k];
    Eigen::VectorXd Tk = Ts[k];
    int np = Tk.size();

    if(Xk.size()==0){
      for(int i=0; i<np; i++){
        CR.row(id) = (1.0-Tk(i))*Ck.row(0)+Tk(i)*Ck.row(1);
        id++;
      }
    }
    else{
      Eigen::VectorXd Xn(Xk.size()+2);
      Xn.segment(1,Xk.size())=Xk;
      Xn(0)=0;Xn(Xn.size()-1)=1;
      for(int i=0; i<np; i++){
        double t=Tk(i);
        for(int j=0; j<Xn.size(); j++){
          if(t>=Xn(j) && t<=Xn(j+1)){
            double tn = (t-Xn(j))/(Xn(j+1)-Xn(j));
            CR.row(id)=(1.0-tn)*Ck.row(j)+tn*Ck.row(j+1);
            id++;
            break;
          }
        }
      }
    }
  }
}


void infinite::diffcurv_color(const MatX3d_list &Cs,
                    const VecXd_list &Ts,
                    const Eigen::VectorXd &T,
                    Eigen::MatrixX3d &CR)
{
  int np = T.size();
  int nb = Cs.size();

  CR.resize(np*nb,3);

  for(int k=0; k<nb; k++){
    Eigen::MatrixX3d Ck = Cs[k];
    Eigen::VectorXd Tk = Ts[k];

    if(Tk.size()==0){
      for(int i=0; i<np; i++){
        CR.row(np*k+i) = (1-T(i))*Ck.row(0)+T(i)*Ck.row(1);
      }
    }
    else{
      Eigen::VectorXd Tn(Tk.size()+2);
      Tn.segment(1,Tk.size())=Tk;
      Tn(0)=0;Tn(Tn.size()-1)=1;
      for(int i=0; i<np; i++){
        double t=T(i);
        for(int j=0; j<Tn.size(); j++){
          if(t>=Tn(j) && t<=Tn(j+1)){
            double tn = (t-Tn(j))/(Tn(j+1)-Tn(j));
            CR.row(np*k+i)=(1-tn)*Ck.row(j)+tn*Ck.row(j+1);
          }
        }
      }
    }
  }
}