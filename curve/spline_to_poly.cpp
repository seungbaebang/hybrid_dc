#include "spline_to_poly.h"

void cubic_split(const Mat42 &C,double t,Mat42 &C1,Mat42 &C2)
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


bool cubic_is_flat(const Mat42 &C, const double &tol)
{
    double ux = std::pow(3.0*C(1,0) - 2.0*C(0,0) - C(3,0),2);
    double uy = std::pow(3.0*C(1,1) - 2.0*C(0,1) - C(3,1),2);
    double vx = std::pow(3.0*C(2,0) - 2.0*C(3,0) - C(0,0),2);
    double vy = std::pow(3.0*C(2,1) - 2.0*C(3,1) - C(0,1),2);
    if(ux<vx)
        ux = vx;
    if(uy<vy)
        uy = vy;
    double tolerance = 16*std::pow(tol,2);

    if((ux+uy) <= tolerance)
        return true;
    else
        return false;
}


void cubic_flat_eval(const Mat42 &C, const double &tol, Eigen::MatrixXd &P, Eigen::VectorXd &T)
{
    if(cubic_is_flat(C,tol)){
        P.resize(2,2);
        P.row(0) = C.row(0);
        P.row(1) = C.row(3);
        T.resize(2);
        T(0) = 0.; T(1) = 1.;
    }
    else{
        Mat42 C1, C2;
        Eigen::MatrixXd P1, P2;
        Eigen::VectorXd T1, T2;
        cubic_split(C,0.5,C1,C2);
        cubic_flat_eval(C1,tol,P1,T1);
        cubic_flat_eval(C2,tol,P2,T2);
        P.resize(P1.rows()+P2.rows()-1,2);
        P.block(0,0,P1.rows(),2)=P1;
        P.block(P1.rows(),0,P2.rows()-1,2)=P2.block(1,0,P2.rows()-1,2);
        T.resize(T1.size()+T2.size()-1);
        T.head(T1.size()) = 0.5*T1;
        T.tail(T2.size()-1) = 0.5*T2.tail(T2.size()-1).array()+0.5;
    }
}

void spline_to_poly(const Mat42_list &CPs, const double &tol, VecXd_list &T_list)
{
    T_list.resize(CPs.size());
    for(int i=0; i<CPs.size(); i++){
        Eigen::MatrixXd Pi;
        Eigen::VectorXd Ti;
        cubic_flat_eval(CPs[i],tol,Pi,Ti);
        T_list[i] = Ti;
    }
}

// void spline_to_poly(const Mat42_list &CPs, const double &tol, Eigen::MatrixXd &P, Eigen::VectorXd &T)
// {
   
//     for(int i=0; i<CPs.size(); i++){
//         Eigen::MatrixXd Pi;
//         Eigen::VectorXd Ti;
//         cubic_flat_eval(CPs[i],tol,Pi,Ti);
//         Eigen::MatrixXd Ptemp(P.rows()+Pi.rows());
//         Ptemp.block(P.rows(),0,Pi.rows(),2)=Pi;
//         Eigen::VectorXd Ttemp(T.size()+Ti.size());
//         Ttemp.segment(T.size(),Ti.size()) = Ti;
//     }
// }