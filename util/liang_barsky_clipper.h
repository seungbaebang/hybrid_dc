#ifndef LIANG_BARSKY_CLIPPER_H
#define LIANG_BARSKY_CLIPPER_H

#include <Eigen/Core>

bool liang_barsky_clipper(double xmin, double ymin, double xmax, double ymax,
                          double x1, double y1, double x2, double y2, double &len);

bool liang_barsky_clipper(const Eigen::Matrix2d &bnd, 
                          const Eigen::RowVector2d &pnt1, 
                          const Eigen::RowVector2d &pnt2,
                          Eigen::RowVector2d &npnt1,
                          Eigen::RowVector2d &npnt2,
                          double &len);

bool liang_barsky_clipper(const Eigen::Matrix2d &bnd, 
                          const Eigen::RowVector2d &pnt1, 
                          const Eigen::RowVector2d &pnt2,
                          Eigen::RowVector2d &npnt1,
                          Eigen::RowVector2d &npnt2,
                          Eigen::RowVector2d &rns);

#endif