#ifndef SAVE_PNG_H
#define SAVE_PNG_H

#include <Eigen/Core>
#include <igl/png/writePNG.h>

void save_png_square(std::string png_path, 
              const unsigned int& width, 
              const unsigned int& height, 
              const Eigen::MatrixXd& RGB);
              
void save_png(std::string png_path, 
              const unsigned int& width, 
              const unsigned int& height, 
              const Eigen::MatrixXd& RGB);

#endif
