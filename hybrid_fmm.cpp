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
#include <igl/setdiff.h>

// #include <GLUT/glut.h>
// #include <cstdlib>

#include "fmm/eval_fmm.h"
#include "fmm/sol_fmm.h"
#include "fmm/quadtree.h"
#include "bem/green_line_integral.h"
#include "bem/singular_on_bnd.h"
#include "curve/subdivide_curves.h"
#include "curve/read_diffcurv_data.h"
#include "curve/diffcurv_color.h"
#include "curve/diffcurv.h"
#include "biem/interpolate_density.h"
#include "util/save_png.h"
#include "util/create_regular_grid.h"


// #define omp_get_thread_num() 0
// set OMP_NUM_THREADS=1

int main( int argc, char **argv )
{
// omp_set_num_threads(1);

  std::string result_folder = "../result/";
  std::string diffcurv_file_path = "/Users/seungbae/Sources/GitHub/infinite_vector_graphics/hybrid_dc_code/data/purple_cloak.txt";

  if(argc>1){
    diffcurv_file_path = std::string(argv[1]);
  }
  std::filesystem::path p(diffcurv_file_path);
  std::string model_name = p.stem();
  std::cout<<"model_name: "<<model_name<<std::endl;
  std::string save_folder = result_folder;// + model_name;

  double image_width;
  double image_height;
  Mat42_list CPs; //nb*(4x2) list of control points of cubic curves
  MatX3d_list CLCs; //pre-defined color values on cubic curves (left side)
  VecXd_list CLTs; //pre-defined parameter values on cubic curves (left side)
  MatX3d_list CRCs; //pre-defined color values on cubic curves (right side)
  VecXd_list CRTs; //pre-defined parameter values on cubic curves (right side)

  infinite::read_diffcurv_data(diffcurv_file_path,image_width,image_height,CPs,CLCs,CLTs,CRCs,CRTs);

  Eigen::MatrixXd RGB; //output rgb values


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

  //get pixel positions
  Eigen::MatrixX2d Q = create_regular_grid(resol_width,resol_height,image_width,image_height);

  int nb = CPs.size(); //number of curves
  int ng = 6;//4; //number of quadrature points
  int ns = 40;//20; //number of line segments for solving stage

  std::cout<<"num curve: "<<nb<<std::endl;

  //parameter for gmres & fmm
  int num_expansion = 4;
  int min_pnt_num = 5;
  int max_depth = std::numeric_limits<int>::max();
  int restart = 40;
  double tol = 2e-2;
  int maxIters = 40;

  double pixel_length = std::sqrt((image_width*image_height)/(double)((resol_width/8)*(resol_height/8)));

  //legendre gauss quadrature parameters and weights
  Eigen::VectorXd tg,wg;
  infinite::legendre_gauss_quadrature(ng,0,1,tg,wg);

  //parameter value for line segments
  Eigen::VectorXd ts = Eigen::VectorXd::LinSpaced(ns+1,0,1);
  Eigen::VectorXd tsc = (ts.head(ns)+ts.tail(ns))/2;
  
  Eigen::MatrixX2d Ps; //(nb*(ns+1)) number of end points line segments 
  Eigen::MatrixX2d Ns; //(nb*ns) number of normal of line segments
  Eigen::MatrixX2d Cs; //(nb*ns) number of mid points of line segments
  Eigen::MatrixX2d Cg; //(nb*ng) number of quadrature points
  Eigen::MatrixX2i Es; //(nb*ns) number of line indices
  Eigen::VectorXd Ls; //(nb*ns) number of length of each line segments

  Eigen::MatrixX3d ug_l, ug_r; //(nb*ng) Dirichlet color boundary value on left, right

  infinite::discretize_diffcurv(CPs,CLCs,CLTs,CRCs,CRTs,ts,tsc,tg,
                                Ps,Cs,Cg,Ns,Es,Ls,ug_l,ug_r);

  Eigen::MatrixXd sigma_g; //(nb*ng) density value ate quadrature points

  //data for quadtree
  std::vector<int> leaf_cells;
  std::vector<std::vector<int > > Q_PI, Q_PEI, Q_QI;
  Eigen::MatrixXi Q_CH;
  Eigen::VectorXi Q_PA, Q_LV;
  Eigen::MatrixXd Q_CN;
  Eigen::VectorXd Q_W;
  Mat2d_list_list Q_PP;
  RowVec2d_list_list Q_LL;

  //data for cell relations
  std::vector<std::vector<int> > Q_adjs, Q_small_seps, Q_big_seps, Q_inters, Q_uni_adjs;
  Eigen::VectorXi qI; //cell index that contains query points

  //data for expansions (cell dependent)
  std::vector<VecXcd_list> Ik_out_list_list;
  std::vector<VecXcd_list> Ok_inter_list_list;
  MatXcd_list Ik_child_list;
  std::vector<std::vector<VecXcd_list> > Ok_inc_list_list;
  VecXcd_list lw_list;

  //data for expansions (query dependent)
  std::vector<RowVecXd_list> G_list;
  std::vector<RowVecXd_list> F_list;
  std::vector<VecXi_list> nI_list;
  std::vector<VecXcd_list> Ok_l3_list;
  VecXcd_list Ik_inc_list;

  Eigen::MatrixXd mu = ug_l-ug_r;
  Eigen::MatrixXd b = (ug_l+ug_r)/2;
  Eigen::VectorXd b_norm = b.colwise().norm();

  Eigen::MatrixXd CPl(CPs.size()*4,2);
  for(int k=0; k<CPs.size(); k++){
    CPl.block(4*k,0,4,2)=CPs[k];
  }

  Eigen::Matrix2d minmaxQ;
  minmaxQ.row(0) = CPl.colwise().minCoeff();
  minmaxQ.row(1) = CPl.colwise().maxCoeff();

  Eigen::Matrix2d bndP;
  bndP<<0,0,image_width,image_height;
  
  double sol_s = igl::get_seconds();

  sigma_g = Eigen::MatrixXd::Ones(b.rows(),b.cols());
  infinite::solve_fmm_gmres_hybrid(Ps,Es,Cs,Ns,Ls,Cg,minmaxQ,b,b_norm,mu,ts,tsc,tg,num_expansion,
    min_pnt_num,max_depth,restart,tol,maxIters,qI,Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,Q_PA,Q_CH,Q_LV,
    Q_CN,Q_W,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_uni_adjs,leaf_cells,Ik_out_list_list,
    Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,lw_list,G_list,F_list,nI_list,Ok_l3_list,
    Ik_inc_list,sigma_g);
  
  double sol_t = igl::get_seconds();
  std::cout<<"sol time: "<<sol_t-sol_s<<std::endl;

  //////////////////////////////////

  //length of each cubic curve
  Eigen::VectorXd cL = infinite::bezier_length(CPs);
  {
    Eigen::VectorXd cLi = cL.array()/10;
    // Eigen::VectorXi nel = cLi.cast<int>().array()+20;

    Eigen::VectorXi nel = ns*Eigen::VectorXi::Ones(nb);

    //list of parameter values line segments for evaluation
    VecXd_list tel(nb);
    VecXd_list tecl(nb);
    for(int k=0; k<nel.size(); k++){
      int ne = nel(k);
      Eigen::VectorXd xe = Eigen::VectorXd::LinSpaced(ne+1,0,1);
      Eigen::VectorXd xec = (xe.head(ne)+xe.tail(ne))/2;
      tel[k]=xe;
      tecl[k]=xec;
    }
    Eigen::MatrixX2d Pe; //end points of line segments for evaluation
    Eigen::MatrixX2d Ne; //normal of line segments for evaluation
    Eigen::MatrixX2d Ce; //mid points of line segments for evaluation
    Eigen::MatrixX2i Ee; //indices of line segments for evaluation
    Eigen::VectorXd Le; //length of line segments for evaluation
    Eigen::MatrixX3d ue_l,ue_r; // color value on line segments on left, right for evaluation

    infinite::discretize_diffcurv(CPs,CLCs,CLTs,CRCs,CRTs,nel,tel,tecl,
                                  Pe,Ce,Ne,Ee,Le,ue_l,ue_r);



    Eigen::MatrixXd sigma_e = infinite::interpolate_density(tg,tecl,sigma_g);
    Eigen::MatrixXd mu_e = ue_l-ue_r;

    double eval_s = igl::get_seconds();
    infinite::eval_fmm_integral(Pe,Ee,Ce,Ne,Le,Q,sigma_e,mu_e,num_expansion,min_pnt_num,max_depth,RGB);
    RGB = RGB.cwiseMax(0);
    RGB = RGB.cwiseMin(1);
    double eval_t = igl::get_seconds();
    

    std::cout<<"eval time: "<<eval_t-eval_s<<std::endl;
   
    if(!std::filesystem::is_directory(save_folder)){
      std::filesystem::create_directories(save_folder);
    }
    std::string png_file = save_folder+"/"+model_name+"_hybrid_fmm.png";
    save_png(png_file,resol_width,resol_height,RGB);
    std::cout<<"png_file : "<<png_file<<std::endl;
  }


  return 0;
}