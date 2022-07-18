#include <vector>
#include <set>
#include <algorithm>
#include <iostream>

#include <filesystem>

#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include <igl/repmat.h>
#include <igl/PI.h>
#include <igl/slice.h>
#include <igl/barycenter.h>
#include <igl/unique.h>
#include <igl/get_seconds.h>
#include <igl/parula.h>

#include <cstdlib>

#include "fmm/sol_fmm.h"
#include "fmm/eval_fmm.h"
#include "bem/green_line_integral.h"
#include "bem/singular_on_bnd.h"
#include "curve/read_diffcurv_data.h"
#include "curve/diffcurv_color.h"
#include "curve/bezier_curves.h"
#include "curve/diffcurv.h"
#include "util/save_png.h"
#include "util/create_regular_grid.h"

Eigen::MatrixXd RGB;

int main( int argc, char **argv )
{


  std::string result_folder = "../result/";
  std::string diffcurv_file_path = "../data/test7_new.txt";
  if(argc>1){
    diffcurv_file_path = std::string(argv[1]);
  }
  std::filesystem::path p(diffcurv_file_path);
  std::string model_name = p.stem();
  std::cout<<"model_name: "<<model_name<<std::endl;

  double image_width;
  double image_height;
  Mat42_list CPs; //nb*(4x2) list of control points of cubic curves
  MatX3d_list CLCs; //pre-defined color values on cubic curves (left side)
  VecXd_list CLTs; //pre-defined parameter values on cubic curves (left side)
  MatX3d_list CRCs; //pre-defined color values on cubic curves (right side)
  VecXd_list CRTs; //pre-defined parameter values on cubic curves (right side)

  Eigen::MatrixX2d CP;
  Eigen::MatrixX4i CE;
  Eigen::VectorXi CI;
  MatX3d_list CLCl;
  VecXd_list CLTl;
  MatX3d_list CRCl;
  VecXd_list CRTl;

  infinite::read_diffcurv_data(diffcurv_file_path,image_width,image_height,CP,CE,CI,CLCl,CLTl,CRCl,CRTl);
  infinite::segment_diffcurv(CP,CE,CI,CLCl,CLTl,CRCl,CRTl,CPs,CLCs,CLTs,CRCs,CRTs);

  // infinite::read_diffcurv_data(diffcurv_file_path,image_width,image_height,CPs,CLCs,CLTs,CRCs,CRTs);

  unsigned int resol_width=(int)image_width;
  unsigned int resol_height=(int)image_height;

  if(argc==3){
    resol_width *= std::atoi(argv[2]);
    resol_height *= std::atoi(argv[2]);
  }
  if(argc==4){
    resol_width=std::atoi(argv[2]);
    resol_height=std::atoi(argv[3]);
  }
  std::cout<<"read done"<<std::endl;

  //get pixel positions
  Eigen::MatrixX2d Q = create_regular_grid(resol_width,resol_height,image_width,image_height);

  int nb = CPs.size(); //number of curves
  int ne = 20; //number of line segments 

  std::cout<<"num cubic bezier curve: "<<nb<<std::endl;

  ///////////////// solve stage begins /////////////////
  
  Eigen::VectorXd te = Eigen::VectorXd::LinSpaced(ne+1,0,1);
  Eigen::VectorXd tec = (te.head(ne)+te.tail(ne))/2;

  Eigen::MatrixX2d P; //(nb*(ne+1)) number of end points line segments 
  Eigen::MatrixX2d N; //(nb*ne) number of normal of line segments
  Eigen::MatrixX2d C; //(nb*ne) number of mid points of line segments
  Eigen::MatrixX2i E; //(nb*ne) number of line indices
  Eigen::VectorXd L; //(nb*ne) number of length of each line segments
  Eigen::MatrixX3d ue_l, ue_r;  //(nb*ne) Dirichlet color boundary value on left, right

  infinite::discretize_diffcurv(CPs,CLCs,CLTs,CRCs,CRTs,te,tec,P,C,N,E,L,ue_l,ue_r);

  Eigen::MatrixXd mu = ue_l-ue_r;

  double sol_s = igl::get_seconds();
  Eigen::MatrixXd sigma; //(nb*ne) density value at mid point of line segments
  {
    Eigen::MatrixXd Gs,Fs;
    infinite::green_line_integral(P,E,N,L,C,Gs,Fs);
    infinite::singular_on_bnd(te,tec,L,Gs,Fs);
    Eigen::MatrixXd b = Fs*mu - 0.5*(ue_l+ue_r);
    sigma = -Gs.householderQr().solve(b);
  }
  double sol_t = igl::get_seconds();
  std::cout<<"sol time: "<<sol_t-sol_s<<std::endl;

  ///////////////// solve stage ends /////////////////
  ///////////////// evaluation stage begins /////////////////

  double eval_s = igl::get_seconds();
  {
    Eigen::MatrixXd Gv,Fv;
    infinite::green_line_integral(P,E,N,L,Q,Gv,Fv);
    RGB = Gv*sigma+Fv*mu;
    RGB = RGB.cwiseMax(0);
    RGB = RGB.cwiseMin(1);

  }
  double eval_t = igl::get_seconds();
  std::cout<<"eval time: "<<eval_t-eval_s<<std::endl;
  ///////////////// evaluation stage ends /////////////////

  std::string png_file=result_folder+model_name+"_bem_"+
                      std::to_string(resol_width)+"_"+std::to_string(resol_height)+"_brute.png";
  save_png(png_file,resol_width,resol_height,RGB);
  std::cout<<"png_file : "<<png_file<<std::endl;
  
  return 0;
}