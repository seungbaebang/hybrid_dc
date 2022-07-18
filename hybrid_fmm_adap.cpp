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
#include "util/read_json.h"




int main( int argc, char **argv )
{
  std::string json_file_path = "../data/cerise_test.json";
  std::string diffcurv_file_path;
  unsigned int resol_height;
  unsigned int resol_width;
  Eigen::Matrix2d zoom_bnd;
  int num_frame;




  std::string result_folder = "../result/";
  // std::string diffcurv_file_path = "../data/zephyr.txt";

  if(argc>1){
    json_file_path = std::string(argv[1]);
  }


  read_json(json_file_path,diffcurv_file_path,resol_height,resol_width,zoom_bnd,num_frame);


  std::filesystem::path p(diffcurv_file_path);
  std::string model_name = p.stem();
  std::cout<<"model_name: "<<model_name<<std::endl;
  std::string save_folder = result_folder + model_name+"_zoom";

  double image_width;
  double image_height;
  Mat42_list CPs; //nb*(4x2) list of control points of cubic curves
  MatX3d_list CLCs; //pre-defined color values on cubic curves (left side)
  VecXd_list CLTs; //pre-defined parameter values on cubic curves (left side)
  MatX3d_list CRCs; //pre-defined color values on cubic curves (right side)
  VecXd_list CRTs; //pre-defined parameter values on cubic curves (right side)

  infinite::read_diffcurv_data(diffcurv_file_path,image_width,image_height,CPs,CLCs,CLTs,CRCs,CRTs);

  Eigen::MatrixXd RGB; //output rgb values

  if(argc==3){
    resol_width *= std::atoi(argv[2]);
    resol_height *= std::atoi(argv[2]);
  }
  if(argc==4){
    resol_width=std::atoi(argv[2]);
    resol_height=std::atoi(argv[3]);
  }


  Eigen::Matrix2d bndP1, bndP2;

  bool x_dominant = true;
  if(image_height>image_width)
      x_dominant = false;

  if(x_dominant){
    double dd = (image_width - image_height)/2;
    bndP1<<0,-dd, image_width, image_height+dd;
  }
  else{
    double dd = (image_height - image_width)/2;
    bndP1<<-dd,0, image_width+dd, image_height;
  }
  bndP2 = zoom_bnd;

  int nb = CPs.size(); //number of curves
  int ng = 4; //number of quadrature points
  int ns = 20; //number of line segments for solving stage

  //parameter for gmres & fmm
  int num_expansion = 4;
  int min_pnt_num = 5;
  int max_depth = std::numeric_limits<int>::max();
  int restart = 60;
  double tol = 1e-2;
  int maxIters = 60;


  // int num_frame = 140;
  std::vector<Eigen::Matrix2d> bndP_list(num_frame);
  std::vector<Eigen::MatrixX2d> Q_list(num_frame);
  Eigen::MatrixX2i resol_list(num_frame,2);
  unsigned int sq_resol=2048;
  resol_width = sq_resol; resol_height = sq_resol;
  {
    

    for(int i=0; i<num_frame; i++){
      double tt = ((double)(num_frame-i-1))/((double)(num_frame-1));
      double fts = tt*tt;
      Eigen::Matrix2d bndP = bndP1*fts + bndP2*(1.0-fts);
      if(bndP(0,0)<0)
        bndP(0,0)=0;
      if(bndP(1,0)>image_width)
        bndP(1,0)=image_width;
      if(bndP(0,1)<0)
        bndP(0,1)=0;
      if(bndP(1,1)>image_height)
        bndP(1,1)=image_height;

      if(image_width==image_height){
        resol_width=sq_resol;
        resol_height=sq_resol;
      }
      else{
        if(x_dominant){
          double len = bndP(1,0)-bndP(0,0);
          double ratio = (double)resol_width/len;
          resol_height = (int)(ratio*(bndP(1,1)-bndP(0,1)));

        }
        else{
          double len = bndP(1,1)-bndP(0,1);
          double ratio = (double)resol_height/len;
          resol_width = (int)(ratio*(bndP(1,0)-bndP(0,0)));
        }
      }
      resol_list(i,0)=resol_width; resol_list(i,1)=resol_height;
      Eigen::MatrixX2d Q = create_regular_grid(resol_width,resol_height,1.0,1.0);
      Q.col(0) = Q.col(0).array()*((bndP(1,0)-bndP(0,0)))+bndP(0,0);
      Q.col(1) = Q.col(1).array()*((bndP(1,1)-bndP(0,1)))+bndP(0,1);
      Q_list[i] = Q;
      bndP_list[i]=bndP;
    }
  }


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

  Eigen::MatrixXd CPl(CPs.size()*4,3);
  for(int k=0; k<CPs.size(); k++){
    CPl.block(4*k,0,4,3)=CPs[k];
  }

  Eigen::Matrix2d minmaxQ;
  minmaxQ.row(0) = CPl.colwise().minCoeff();
  minmaxQ.row(1) = CPl.colwise().maxCoeff();

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
    Eigen::VectorXi nel = cLi.cast<int>().array()+20;

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
    Eigen::MatrixX2d Q = Q_list[0];

    double eval_s = igl::get_seconds();
    infinite::eval_fmm_integral(Pe,Ee,Ce,Ne,Le,Q,sigma_e,mu_e,num_expansion,min_pnt_num,max_depth,RGB);
    RGB = RGB.cwiseMax(0);
    RGB = RGB.cwiseMin(1);
    double eval_t = igl::get_seconds();

    std::cout<<"eval time: "<<eval_t-eval_s<<std::endl;
   
    if(!std::filesystem::is_directory(save_folder)){
      std::filesystem::create_directories(save_folder);
    }
    std::string png_file = save_folder+"/"+model_name+"_hybrid_frame_0.png";
    save_png_square(png_file,resol_list(0,0),resol_list(0,1),RGB);
    std::cout<<"png_file : "<<png_file<<std::endl;
  }

  ///////////////// zoom begins /////////////////

  for(int frame=0; frame<num_frame; frame++){
    std::cout<<"frame: "<<frame<<std::endl;
    Eigen::Matrix2d bndP = bndP_list[frame];
  
    Eigen::MatrixX2d Q = Q_list[frame];
    double pixel_length = std::sqrt(((bndP(1,0)-bndP(0,0))*(bndP(1,1)-bndP(0,1)))
                          /(double)((resol_list(frame,0)/8)*(resol_list(frame,1)/8)));

    std::cout<<"pixel_length: "<<pixel_length<<std::endl;

    ///////////////// adaptive subdivision begins /////////////////

    for(int adap_iter=1; adap_iter<30; adap_iter++){
      std::cout<<"adap_iter: "<<adap_iter<<std::endl;

      Eigen::VectorXi cI(Es.rows());
      #pragma omp parallel for
      for(int i=0; i<leaf_cells.size(); i++){
        for(int ci=0; ci<Q_PI[leaf_cells[i]].size(); ci++){
          cI(Q_PI[leaf_cells[i]][ci]) = leaf_cells[i];
        }
      }
      std::vector<std::vector<int> > cEI(Es.rows());
      // #pragma omp parallel for
      for(int i=0; i<leaf_cells.size(); i++){
        for(int ci=0; ci<Q_PEI[leaf_cells[i]].size(); ci++){
          cEI[Q_PEI[leaf_cells[i]][ci]].emplace_back(leaf_cells[i]);
        }
      }


      std::vector<int> in_list;
      for(int k=0; k<nb; k++){
        for(int i=0; i<ns; i++){
          int ei0 = Es(ns*k+i,0);
          int ei1 = Es(ns*k+i,1);
          if( (Ps(ei0,0)>bndP(0,0) && Ps(ei0,0)<bndP(1,0) && 
          Ps(ei0,1)>bndP(0,1) && Ps(ei0,1)<bndP(1,1)) ||
          (Ps(ei1,0)>bndP(0,0) && Ps(ei1,0)<bndP(1,0) && 
          Ps(ei1,1)>bndP(0,1) && Ps(ei1,1)<bndP(1,1)) )
          {
            in_list.emplace_back(k);
            break;
          }
        }
      }
      Eigen::VectorXi inI = Eigen::Map<Eigen::VectorXi>(in_list.data(),in_list.size());
      Eigen::VectorXi in_qids(ng*inI.size());
      for(int i=0; i<inI.size(); i++){
        for(int j=0; j<ng; j++){
          in_qids(ng*i+j) = ng*inI(i)+j;
        }
      }

      Eigen::VectorXd cL_in = igl::slice(cL,inI);
      Eigen::MatrixXd sigma_g_in = igl::slice(sigma_g,in_qids,1);

      //indices fof curve that needs subdivision
      Eigen::VectorXi subI_in = infinite::get_adap_subdivision_list(cL_in,tg,sigma_g_in,pixel_length);
      if(subI_in.size()<1){
        std::cout<<"stop adap subdivision!"<<std::endl;
        break;
      }
      Eigen::VectorXi subI;
      igl::slice(inI,subI_in,subI);

      Eigen::VectorXi allI = Eigen::VectorXi::LinSpaced(nb,0,nb-1);
      Eigen::VectorXi origI, IA;
      igl::setdiff(allI,subI,origI,IA);

      double sub_t = 0.5;
      //split diffusion curve in half for subdivision curves
      infinite::rediscretize_diffcurv(ts,tsc,tg,subI,origI,sub_t, 
              CPs,CLCs,CLTs,CRCs,CRTs,Ps,Cs,Cg,Ns,Es,Ls,ug_l,ug_r);

      mu = ug_l-ug_r;
      b = (ug_l+ug_r)/2.0;
      b_norm = b.colwise().norm();

      Eigen::VectorXi solI, newI;
      infinite::resolve_fmm_gmres_hybrid_local(Ps,Es,Cs,Ns,Ls,Cg,b,b_norm,mu,ts,tsc,tg,num_expansion,
        min_pnt_num,max_depth,restart,tol,maxIters,cI,cEI,subI,origI,sub_t,bndP,qI,Q_PI,Q_PEI,Q_QI,Q_LL,Q_PP,
        Q_PA,Q_CH,Q_LV,Q_CN,Q_W,Q_adjs,Q_small_seps,Q_inters,Q_big_seps,Q_uni_adjs,leaf_cells,
        Ik_out_list_list,Ok_inter_list_list,Ik_child_list,Ok_inc_list_list,lw_list,G_list,F_list,
        nI_list,Ok_l3_list,Ik_inc_list,sigma_g,solI,newI);
      {
        nb = CPs.size();
        cL = infinite::bezier_length(CPs);
        Eigen::VectorXd cLi = cL.array()/10;
        Eigen::VectorXi nel = cLi.cast<int>().array()+20;

        VecXd_list tel(nb);
        VecXd_list tecl(nb);
        for(int k=0; k<nel.size(); k++){
          int ne = nel(k);
          Eigen::VectorXd xe = Eigen::VectorXd::LinSpaced(ne+1,0,1);
          Eigen::VectorXd xec = (xe.head(ne)+xe.tail(ne))/2;
          tel[k]=xe;
          tecl[k]=xec;
        }

        Eigen::MatrixX2d Pe,Ne,Ce;
        Eigen::MatrixX2i Ee;
        Eigen::VectorXd Le;
        Eigen::MatrixX3d ue_l, ue_r;

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

        std::string png_file = save_folder+"/"+model_name+"_hybrid_frame_"+std::to_string(frame)
        +"_adap_"+std::to_string(adap_iter)+".png";
        save_png_square(png_file,resol_list(frame,0),resol_list(frame,1),RGB);       
        std::cout<<"png_file : "<<png_file<<std::endl;
      }
    }
    std::cout<<"stop adap subdivision!"<<std::endl;

    {
      nb = CPs.size();
      cL = infinite::bezier_length(CPs);
      Eigen::VectorXd cLi = cL.array()/10;
      Eigen::VectorXi nel = cLi.cast<int>().array()+20;

      VecXd_list tel(nb);
      VecXd_list tecl(nb);
      for(int k=0; k<nel.size(); k++){
        int ne = nel(k);
        Eigen::VectorXd xe = Eigen::VectorXd::LinSpaced(ne+1,0,1);
        Eigen::VectorXd xec = (xe.head(ne)+xe.tail(ne))/2;
        tel[k]=xe;
        tecl[k]=xec;
      }

      Eigen::MatrixX2d Pe,Ne,Ce;
      Eigen::MatrixX2i Ee;
      Eigen::VectorXd Le;
      Eigen::MatrixX3d ue_l, ue_r;

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

      std::string png_file = save_folder+"/"+model_name+"_hybrid_frame_"+std::to_string(frame)+".png";
      // save_png(png_file,resol_width,resol_height,RGB);   
      save_png_square(png_file,resol_list(frame,0),resol_list(frame,1),RGB);      
      std::cout<<"png_file : "<<png_file<<std::endl;
    }


    // std::string save_file = save_folder+"/"+model_name+"_sub_fin.txt";
    // infinite::save_diffcurv_data(save_file,image_width,image_height,CPs,CLCs,CLTs,CRCs,CRTs);

  }


  return 0;
}