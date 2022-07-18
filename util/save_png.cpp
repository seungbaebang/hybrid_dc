#include "save_png.h"
#include <igl/writeDMAT.h>


void save_png_square(std::string png_path, 
              const unsigned int& width, 
              const unsigned int& height, 
              const Eigen::MatrixXd& RGB)
{
  unsigned int big, small;
  if(width>height){
    big = width;
    small = height;
  }
  else{
    big = height;
    small = width;
  }
  unsigned int si = std::round(big-small)/2;

  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(big,big);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(big,big);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(big,big);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(big,big);

  R.setZero(); G.setZero(); B.setZero(); A.setZero();

  
  int k=0;
  for(int y=0; y<height; ++y)
  {
    for(int x=0; x<width; ++x)
    {
      int xi, yi;
      if(width>height){
        xi = x; yi = y + si;
      }
      else{
        xi= x + si; yi = y;
      }
      
      R(xi,yi) = (unsigned char)(255*RGB(k,0));
      G(xi,yi) = (unsigned char)(255*RGB(k,1));
      B(xi,yi) = (unsigned char)(255*RGB(k,2));
      A(xi,yi) = (unsigned char)(255);
      k++;
    }
  }
  igl::png::writePNG(R,G,B,A,png_path);
}




void save_png(std::string png_path, 
              const unsigned int& width, 
              const unsigned int& height, 
              const Eigen::MatrixXd& RGB)
{
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(width,height);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(width,height);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(width,height);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(width,height);

  // igl::writeDMAT(png_path+".dmat",RGB);
  
  int k=0;
  for(int y=0; y<height; ++y)
  {
    for(int x=0; x<width; ++x)
    {
      R(x,y) = (unsigned char)(255*RGB(k,0));
      G(x,y) = (unsigned char)(255*RGB(k,1));
      B(x,y) = (unsigned char)(255*RGB(k,2));
      A(x,y) = (unsigned char)(255);
      k++;
    }
  }
  igl::png::writePNG(R,G,B,A,png_path);
}
