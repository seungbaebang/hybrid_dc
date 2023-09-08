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
#include "util/read_json.h"

Eigen::MatrixXd RGB;

int main( int argc, char **argv )
{

  std::string json_file_path = "../data/cerise_test.json";

  std::string diffcurv_file_path;
  unsigned int resol_height;
  unsigned int resol_width;
  Eigen::Matrix2d zoom_bnd;
  int num_frame;

  std::string result_folder = "../result/";

  if(argc>1){
    json_file_path = std::string(argv[1]);
  }


  read_json(json_file_path,diffcurv_file_path,resol_height,resol_width,zoom_bnd,num_frame);


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

  // Eigen::MatrixX2d CP;
  // Eigen::MatrixX4i CE;
  // Eigen::VectorXi CI;
  // MatX3d_list CLCl;
  // VecXd_list CLTl;
  // MatX3d_list CRCl;
  // VecXd_list CRTl;

  // infinite::read_diffcurv_data(diffcurv_file_path,image_width,image_height,CP,CE,CI,CLCl,CLTl,CRCl,CRTl);
  // infinite::segment_diffcurv(CP,CE,CI,CLCl,CLTl,CRCl,CRTl,CPs,CLCs,CLTs,CRCs,CRTs);

  infinite::read_diffcurv_data(diffcurv_file_path,image_width,image_height,CPs,CLCs,CLTs,CRCs,CRTs);

  // unsigned int resol_width=(int)image_width;
  // unsigned int resol_height=(int)image_height;

  // if(argc==3){
  //   resol_width *= std::atoi(argv[2]);
  //   resol_height *= std::atoi(argv[2]);
  // }
  // if(argc==4){
  //   resol_width=std::atoi(argv[2]);
  //   resol_height=std::atoi(argv[3]);
  // }
  std::cout<<"read done"<<std::endl;

  //get pixel positions
  Eigen::MatrixX2d Q = create_regular_grid(resol_width,resol_height,1.0,1.0);
  Q.col(0) = Q.col(0).array()*((zoom_bnd(1,0)-zoom_bnd(0,0)))+zoom_bnd(0,0);
  Q.col(1) = Q.col(1).array()*((zoom_bnd(1,1)-zoom_bnd(0,1)))+zoom_bnd(0,1);

  int nb = CPs.size(); //number of curves
  int ne = 100; //number of line segments 

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

  int num_expansion = 4;
  int min_pnt_num = 5;
  int max_depth = std::numeric_limits<int>::max();
  int restart = 40;
  double tol = 2e-2;
  int maxIters = 40;

  double sol_s = igl::get_seconds();
  Eigen::MatrixXd sigma; //(nb*ne) density value at mid point of line segments
  {
    VecXi_list sing_id_list;
    VecXd_list sing_t_list;
    infinite::singular_on_bnd(te,tec,nb,sing_id_list,sing_t_list);

    Eigen::MatrixXd b = (ue_l+ue_r)/2;
    infinite::solve_fmm_gmres_bem(P,E,C,N,L,mu,b,sing_id_list,sing_t_list,
      num_expansion,min_pnt_num,max_depth,restart,tol,maxIters,sigma); 
  }
  double sol_t = igl::get_seconds();
  std::cout<<"sol time: "<<sol_t-sol_s<<std::endl;

  ///////////////// solve stage ends /////////////////
  ///////////////// evaluation stage begins /////////////////

  double eval_s = igl::get_seconds();
  {
    infinite::eval_fmm_integral(P,E,C,N,L,Q,sigma,mu,num_expansion,min_pnt_num,max_depth,RGB);

    // infinite::eval_fmm_integral_noclip(P,E,C,N,L,Q,sigma,mu,num_expansion,min_pnt_num,4,RGB);
    // infinite::eval_fmm_integral_uni(P,E,C,N,L,Q,sigma,mu,num_expansion,4,RGB);

    RGB = RGB.cwiseMax(0);
    RGB = RGB.cwiseMin(1);

  }
  double eval_t = igl::get_seconds();
  std::cout<<"eval time: "<<eval_t-eval_s<<std::endl;
  ///////////////// evaluation stage ends /////////////////

  std::string png_file=result_folder+model_name+"_bem_ne_"+
                      std::to_string(ne)+".png";
  save_png(png_file,resol_width,resol_height,RGB);
  std::cout<<"png_file : "<<png_file<<std::endl;
  
  return 0;
}