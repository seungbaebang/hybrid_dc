#include "read_diffcurv_data.h"



void infinite::save_diffcurv_data(const std::string &file_name,
                        const double &image_width,
                        const double &image_height,
                        const Eigen::MatrixX2d &CP,
                        const Eigen::MatrixX4i &CE,
                        const Eigen::VectorXi &CI,
                        const MatX3d_list &CLCl,
                        const VecXd_list &CLTl,
                        const MatX3d_list &CRCl,
                        const VecXd_list &CRTl)
{
  FILE *fp = fopen(file_name.c_str(),"wb");

  fprintf(fp, "%lg %lg\n", image_width, image_height);
  fprintf(fp, "%d\n", CP.rows());
  for(int i=0; i<CP.rows(); i++){
    for(int j=0; j<CP.cols(); j++){
      fprintf(fp,"%0.6f\n", CP(i,j));
    }
  }
  fprintf(fp, "%d\n", CE.rows());
  for(int i=0; i<CE.rows(); i++){
    for(int j=0; j<CE.cols(); j++){
      fprintf(fp,"%d\n", CE(i,j));
    }
  }
  for(int i=0; i<CI.rows(); i++){
    fprintf(fp,"%d\n", CI(i));
  }
  fprintf(fp, "%d\n", CLCl.size());
  for(int k=0; k<CLCl.size(); k++){
    fprintf(fp, "%d\n", CLCl[k].rows());
    for(int i=0; i<CLCl[k].rows(); i++){
      for(int j=0; j<CLCl[k].cols(); j++){
        fprintf(fp, "%0.6f\n", CLCl[k](i,j));
      }
    }
    for(int i=0; i<CLTl[k].size(); i++){
      fprintf(fp, "%0.6f\n", CLTl[k](i));
    }
  }
  for(int k=0; k<CRCl.size(); k++){

    fprintf(fp, "%d\n", CRCl[k].rows());
    for(int i=0; i<CRCl[k].rows(); i++){
      for(int j=0; j<CRCl[k].cols(); j++){
        fprintf(fp, "%0.6f\n", CRCl[k](i,j));
      }
    }
    for(int i=0; i<CRTl[k].size(); i++){
      fprintf(fp, "%0.6f\n", CRTl[k](i));
    }
  }
  fclose(fp);
}



void infinite::read_diffcurv_data(const std::string &file_name,
                        double &image_width,
                        double &image_height,
                        Eigen::MatrixX2d &CP,
                        Eigen::MatrixX4i &CE,
                        Eigen::VectorXi &CI,
                        MatX3d_list &CLCl,
                        VecXd_list &CLTl,
                        MatX3d_list &CRCl,
                        VecXd_list &CRTl)
{
  FILE *fp = fopen(file_name.c_str(),"rb");
  
  fscanf(fp,"%lf %lf",&image_width, &image_height);
  int np;
  fscanf(fp,"%d",&np);
  CP.resize(np,2);
  for(int i=0; i<np; i++){
    for(int j=0; j<2; j++){
      double d;
      fscanf(fp,"%lf",&d);
      CP(i,j)=d;
    }
  }

  int ne;
  fscanf(fp,"%d",&ne);
  CE.resize(ne,4);
  for(int i=0; i<ne; i++){
    for(int j=0; j<4; j++){
      int id;
      fscanf(fp,"%d",&id);
      CE(i,j)=id;
    }
  }
  CI.resize(ne);
  for(int i=0; i<ne; i++){
    int id;
    fscanf(fp,"%d",&id);
    CI(i)=id;
  }

  int nb;
  fscanf(fp,"%d",&nb);
  CLCl.resize(nb);
  CLTl.resize(nb);
  CRCl.resize(nb);
  CRTl.resize(nb);
  for(int k=0; k<nb; k++){
    int nt;
    fscanf(fp,"%d",&nt);
    Eigen::MatrixX3d CLC(nt,3);
    Eigen::VectorXd CLT(nt);
    for(int i=0; i<nt; i++){
      for(int j=0; j<3; j++){
        double d;
        fscanf(fp,"%lf",&d);
        CLC(i,j)=d;
      }
    }
    for(int i=0; i<nt; i++){
      double d;
      fscanf(fp,"%lf",&d);
      CLT(i)=d;
    }
    CLCl[k]=CLC;
    CLTl[k]=CLT;
  }
  for(int k=0; k<nb; k++){
    int nt;
    fscanf(fp,"%d",&nt);
    Eigen::MatrixX3d CRC(nt,3);
    Eigen::VectorXd CRT(nt);
    for(int i=0; i<nt; i++){
      for(int j=0; j<3; j++){
        double d;
        fscanf(fp,"%lf",&d);
        CRC(i,j)=d;
      }
    }
    for(int i=0; i<nt; i++){
      double d;
      fscanf(fp,"%lf",&d);
      CRT(i)=d;
    }
    CRCl[k]=CRC;
    CRTl[k]=CRT;
  }
  fclose(fp);
}


void infinite::save_diffcurv_data(const std::string &file_name,
                        double &image_width,
                        double &image_height,
                        Mat42_list &CPs,
                        MatX3d_list &CLCs,
                        VecXd_list &CLTs,
                        MatX3d_list &CRCs,
                        VecXd_list &CRTs)
{
  FILE *fp = fopen(file_name.c_str(),"wb");
  fprintf(fp, "%lg %lg\n", image_width, image_height);
  fprintf(fp, "%d\n", CPs.size());
  for(int k=0; k<CPs.size(); k++){
    for(int i=0; i<CPs[k].rows(); i++){
      for(int j=0; j<CPs[k].cols(); j++){
        fprintf(fp,"%0.17lg\n", CPs[k](i,j));
      }
    }
  }
  for(int k=0; k<CLTs.size(); k++){
    fprintf(fp, "%d\n", CLTs[k].size());
    for(int i=0; i<CLCs[k].rows(); i++){
      for(int j=0; j<CLCs[k].cols(); j++){
        fprintf(fp, "%0.17lg\n", CLCs[k](i,j));
      }
    }
    for(int i=0; i<CLTs[k].size(); i++){
      fprintf(fp, "%0.17lg\n", CLTs[k](i));
    }
  }
  for(int k=0; k<CRTs.size(); k++){
    fprintf(fp, "%d\n", CRTs[k].size());
    for(int i=0; i<CRCs[k].rows(); i++){
      for(int j=0; j<CRCs[k].cols(); j++){
        fprintf(fp, "%0.17lg\n", CRCs[k](i,j));
      }
    }
    for(int i=0; i<CRTs[k].size(); i++){
      fprintf(fp, "%0.17lg\n", CRTs[k](i));
    }
  }
  fclose(fp);
}


void infinite::read_diffcurv_data(const std::string &file_name,
                        double &image_width,
                        double &image_height,
                        Mat42_list &CPs,
                        MatX3d_list &CLCs,
                        VecXd_list &CLTs,
                        MatX3d_list &CRCs,
                        VecXd_list &CRTs)
{
  FILE *fp = fopen(file_name.c_str(),"rb");
  
  fscanf(fp,"%lg %lg",&image_width, &image_height);
  int nb;
  fscanf(fp,"%d",&nb);
  CPs.resize(nb);
  CLCs.resize(nb);
  CLTs.resize(nb);
  CRCs.resize(nb);
  CRTs.resize(nb);
  for(int k=0; k<nb; k++){
    Mat42 CP;
    for(int i=0; i<4; i++){
      for(int j=0; j<2; j++){
        double d;
        fscanf(fp,"%lg",&d);
        CP(i,j)=d;
      }
    }
    CPs[k]=CP;
  }
  for(int k=0; k<nb; k++){
    int nt;
    fscanf(fp,"%d",&nt);
    Eigen::MatrixX3d CLC(nt+2,3);
    Eigen::VectorXd CLT(nt);
    for(int i=0; i<(nt+2); i++){
      for(int j=0; j<3; j++){
        double d;
        fscanf(fp,"%lg",&d);
        CLC(i,j)=d;
      }
    }
    for(int i=0; i<nt; i++){
      double d;
      fscanf(fp,"%lg",&d);
      CLT(i)=d;
    }
    CLCs[k]=CLC;
    CLTs[k]=CLT;
  }
  for(int k=0; k<nb; k++){
    int nt;
    fscanf(fp,"%d",&nt);
    Eigen::MatrixX3d CRC(nt+2,3);
    Eigen::VectorXd CRT(nt);
    for(int i=0; i<(nt+2); i++){
      for(int j=0; j<3; j++){
        double d;
        fscanf(fp,"%lg",&d);
        CRC(i,j)=d;
      }
    }
    for(int i=0; i<nt; i++){
      double d;
      fscanf(fp,"%lg",&d);
      CRT(i)=d;
    }
    CRCs[k]=CRC;
    CRTs[k]=CRT;
  }
  fclose(fp);
}
