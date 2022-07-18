#include "create_regular_grid.h"


Eigen::MatrixX2d create_regular_grid(const int &resol_width,
                                     const int &resol_height,
                                     const double &image_width, 
                                     const double &image_height)
{
  // Eigen::VectorXd xc =Eigen::VectorXd::LinSpaced(resol_width,0,image_width);
  // Eigen::VectorXd yc =Eigen::VectorXd::LinSpaced(resol_height,0,image_height);

  Eigen::VectorXd xe =Eigen::VectorXd::LinSpaced(resol_width+1,0,image_width);
  Eigen::VectorXd ye =Eigen::VectorXd::LinSpaced(resol_height+1,0,image_height);
  Eigen::VectorXd xc = (xe.head(resol_width) + xe.tail(resol_width))/2;
  Eigen::VectorXd yc = (ye.head(resol_height) + ye.tail(resol_height))/2;

  Eigen::MatrixX2d Q(resol_width*resol_height,2);
  int k=0;
  for(int yi=0; yi<resol_height; ++yi)
  {
    for(int xi=0; xi<resol_width; ++xi)
    {
      Q.row(k++)<<xc(xi),yc(yi);
    }
  }
  // Q = Q.array()+1e-4;
  return Q;
}
