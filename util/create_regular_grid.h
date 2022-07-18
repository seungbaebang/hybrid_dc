#ifndef CREATE_REGULAR_GRID_H
#define CREATE_REGULAR_GRID_H

#include <Eigen/Core>

Eigen::MatrixX2d create_regular_grid(const int &resol_width,
                                     const int &resol_height,
                                     const double &image_width, 
                                     const double &image_height);


#endif
