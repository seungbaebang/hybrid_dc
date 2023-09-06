#ifndef SPLINE_TO_POLY_H
#define SPLINE_TO_POLY_H

#include <Eigen/Core>
#include <vector>
#include <iostream>

#include "../util/types.h"


// bool cubic_is_flat(const Mat42 &C, const double &tol);

void spline_to_poly(const Mat42_list &CPs, const double &tol, VecXd_list &T_list);


#endif
