#include "liang_barsky_clipper.h"

bool liang_barsky_clipper(double xmin, double ymin, double xmax, double ymax,
                          double x1, double y1, double x2, double y2, double &len) {
  // defining variables
  double p1 = -(x2 - x1);
  double p2 = -p1;
  double p3 = -(y2 - y1);
  double p4 = -p3;

  double q1 = x1 - xmin;
  double q2 = xmax - x1;
  double q3 = y1 - ymin;
  double q4 = ymax - y1;

  double posarr[5], negarr[5];
  int posind = 1, negind = 1;
  posarr[0] = 1;
  negarr[0] = 0;

  if ((p1 == 0 && q1 < 0) || (p2 == 0 && q2 < 0) || (p3 == 0 && q3 < 0) || (p4 == 0 && q4 < 0)) {
      return false;
  }
  if (p1 != 0) {
    float r1 = q1 / p1;
    float r2 = q2 / p2;
    if (p1 < 0) {
      negarr[negind++] = r1; // for negative p1, add it to negative array
      posarr[posind++] = r2; // and add p2 to positive array
    } else {
      negarr[negind++] = r2;
      posarr[posind++] = r1;
    }
  }
  if (p3 != 0) {
    float r3 = q3 / p3;
    float r4 = q4 / p4;
    if (p3 < 0) {
      negarr[negind++] = r3;
      posarr[posind++] = r4;
    } else {
      negarr[negind++] = r4;
      posarr[posind++] = r3;
    }
  }
  double rn1 = *std::max_element(negarr,negarr+negind);
  double rn2 = *std::min_element(posarr,posarr+posind);


  if (rn1 > rn2)  { // reject
    return false;
  }
  len = rn2-rn1;

  return true;
}


bool liang_barsky_clipper(const Eigen::Matrix2d &bnd, 
                          const Eigen::RowVector2d &pnt1, 
                          const Eigen::RowVector2d &pnt2,
                          Eigen::RowVector2d &npnt1,
                          Eigen::RowVector2d &npnt2,
                          double &len) {
  double xmin = bnd(0,0);
  double xmax = bnd(1,0);
  double ymin = bnd(0,1);
  double ymax = bnd(1,1);
  double x1 = pnt1(0);
  double y1 = pnt1(1);
  double x2 = pnt2(0);
  double y2 = pnt2(1);
  // defining variables
  double p1 = -(x2 - x1);
  double p2 = -p1;
  double p3 = -(y2 - y1);
  double p4 = -p3;

  double q1 = x1 - xmin;
  double q2 = xmax - x1;
  double q3 = y1 - ymin;
  double q4 = ymax - y1;

  double posarr[5], negarr[5];
  int posind = 1, negind = 1;
  posarr[0] = 1;
  negarr[0] = 0;

  if ((p1 == 0 && q1 < 0) || (p2 == 0 && q2 < 0) || (p3 == 0 && q3 < 0) || (p4 == 0 && q4 < 0)) {
      return false;
  }
  if (p1 != 0) {
    float r1 = q1 / p1;
    float r2 = q2 / p2;
    if (p1 < 0) {
      negarr[negind++] = r1; // for negative p1, add it to negative array
      posarr[posind++] = r2; // and add p2 to positive array
    } else {
      negarr[negind++] = r2;
      posarr[posind++] = r1;
    }
  }
  if (p3 != 0) {
    float r3 = q3 / p3;
    float r4 = q4 / p4;
    if (p3 < 0) {
      negarr[negind++] = r3;
      posarr[posind++] = r4;
    } else {
      negarr[negind++] = r4;
      posarr[posind++] = r3;
    }
  }
  double rn1 = *std::max_element(negarr,negarr+negind);
  double rn2 = *std::min_element(posarr,posarr+posind);

  if (rn1 > rn2)  { // reject
    return false;
  }
  len = rn2-rn1;
  npnt1(0) = x1 + p2*rn1;
  npnt1(1) = y1 + p4*rn1;

  npnt2(0) = x1 + p2*rn2;
  npnt2(1) = y1 + p4*rn2;
   
  return true;
}


bool liang_barsky_clipper(const Eigen::Matrix2d &bnd, 
                          const Eigen::RowVector2d &pnt1, 
                          const Eigen::RowVector2d &pnt2,
                          Eigen::RowVector2d &npnt1,
                          Eigen::RowVector2d &npnt2,
                          Eigen::RowVector2d &rns) 
{
  double xmin = bnd(0,0);
  double xmax = bnd(1,0);
  double ymin = bnd(0,1);
  double ymax = bnd(1,1);
  double x1 = pnt1(0);
  double y1 = pnt1(1);
  double x2 = pnt2(0);
  double y2 = pnt2(1);
  // defining variables
  double p1 = -(x2 - x1);
  double p2 = -p1;
  double p3 = -(y2 - y1);
  double p4 = -p3;

  double q1 = x1 - xmin;
  double q2 = xmax - x1;
  double q3 = y1 - ymin;
  double q4 = ymax - y1;

  double posarr[5], negarr[5];
  int posind = 1, negind = 1;
  posarr[0] = 1;
  negarr[0] = 0;

  if ((p1 == 0 && q1 < 0) || (p2 == 0 && q2 < 0) || (p3 == 0 && q3 < 0) || (p4 == 0 && q4 < 0)) {
      return false;
  }
  if (p1 != 0) {
    double r1 = q1 / p1;
    double r2 = q2 / p2;
    if (p1 < 0) {
      negarr[negind++] = r1; // for negative p1, add it to negative array
      posarr[posind++] = r2; // and add p2 to positive array
    } else {
      negarr[negind++] = r2;
      posarr[posind++] = r1;
    }
  }
  if (p3 != 0) {
    double r3 = q3 / p3;
    double r4 = q4 / p4;
    if (p3 < 0) {
      negarr[negind++] = r3;
      posarr[posind++] = r4;
    } else {
      negarr[negind++] = r4;
      posarr[posind++] = r3;
    }
  }
  double rn1 = *std::max_element(negarr,negarr+negind);
  double rn2 = *std::min_element(posarr,posarr+posind);

  if (rn1 > rn2)  { // reject
    return false;
  }
  rns(0) = rn1;
  rns(1) = rn2;
  // len = rn2-rn1;
  npnt1(0) = x1 + p2*rn1;
  npnt1(1) = y1 + p4*rn1;

  npnt2(0) = x1 + p2*rn2;
  npnt2(1) = y1 + p4*rn2;
   
  return true;
}