#include "singular_on_bnd.h"
#include <iostream>


// void infinite::singular_on_bnd(const Mat42_list &CPs, 
//                       const Eigen::VectorXd &XE, 
//                       const Eigen::VectorXd &XG, 
//                       Eigen::VectorXd &TOE)
// {
//   int nb = CPs.size();
//   int ne = XE.size()/nb-1;
//   int ng = XG.size()/nb;

//   TOE.resize(nb*ne);

//   for(int k=0; k<nb; k++){
//     Mat42 CP = CPs[k];
//     Eigen::VectorXd xe = XE.segment((ne+1)*k,ne+1);
//     Eigen::VectorXd xg = XG.segment(ng*k,ng);
//     Eigen::MatrixXd TT(ng,ne);
//     for(int j=0; j<ng; j++){
//       TT.row(j) = Eigen::RowVectorXd((xg(j)-xe.segment(0,ne).array())
//                   /(xe.segment(1,ne).array()-xe.segment(0,ne).array()));

//       // Eigen::VectorXd tt = Eigen::VectorXd((xg(j)-xe.segment(0,ne).array())
//       //             /(xe.segment(1,ne).array()-xe.segment(0,ne).array()));
//     }
//     Eigen::VectorXd cI, null;
//     igl::mat_min(TT,1,null,cI);

//     Eigen::VectorXd xgn = igl::slice(xg,cI);
//     Eigen::MatrixXd TE(ne,2);
//     TE.col(0) = xe.segment(0,ne);
//     TE.col(1) = xgn;

//     // Eigen::VectorXd toe = infinite::cubic_length(CP,TE,30);
//     TOE.segment(ne*k,ne) = infinite::cubic_length(CP,TE,30);
//   }
// }

void infinite::singular_on_bnd(const Eigen::VectorXd &xe, 
                     const Eigen::VectorXd &xc,
                     const int &nb,
                     VecXi_list &sing_id_list,
                     VecXd_list &sing_t_list)
{
  int nc = xc.size();   //number of sample point
  int ne = xe.size()-1; //number of line segment

  Eigen::MatrixXd TT;

  TT.resize(nc,ne);
  for(int j=0; j<nc; j++){
    TT.row(j) = (xc(j)-xe.head(ne).array())/
      (xe.tail(ne).array()-xe.head(ne).array());
  }
  TT.resize(nc,ne);

  sing_id_list.resize(nb*nc);
  sing_t_list.resize(nb*nc);

  std::vector<std::vector<int> > sing_id_vec(nc);
  std::vector<std::vector<double> > sing_t_vec(nc);

  for(int i=0; i<nc; i++){
    for(int j=0; j<ne; j++){
      if(TT(i,j)<=1 && TT(i,j)>=0){
        sing_id_vec[i].emplace_back(j);
        sing_t_vec[i].emplace_back(TT(i,j));
      }
    }
  }
  VecXi_list sing_ids(nc);
  VecXd_list sing_ts(nc);

  for(int i=0; i<nc; i++){
    Eigen::VectorXi sing_id = Eigen::Map<Eigen::VectorXi>(sing_id_vec[i].data(),sing_id_vec[i].size());
    Eigen::VectorXd sing_t = Eigen::Map<Eigen::VectorXd>(sing_t_vec[i].data(),sing_t_vec[i].size());
    sing_ids[i]=sing_id;
    sing_ts[i]=sing_t;
  }
  for(int k=0; k<nb; k++){
    for(int i=0; i<nc; i++){
      sing_id_list[k*nc+i] = sing_ids[i].array()+k*ne;
      sing_t_list[k*nc+i] = sing_ts[i];
    }
  }
}


void infinite::singular_on_bnd(const Eigen::VectorXd &xe, 
                     const Eigen::VectorXd &xc,
                     const int &nb,
                     std::vector<std::vector<int>> &sing_id_list,
                     std::vector<std::vector<double>> &sing_t_list)
{
  int nc = xc.size();   //number of sample point
  int ne = xe.size()-1; //number of line segment

  Eigen::MatrixXd TT;

  TT.resize(nc,ne);
  for(int j=0; j<nc; j++){
    TT.row(j) = (xc(j)-xe.head(ne).array())/
      (xe.tail(ne).array()-xe.head(ne).array());
  }
  TT.resize(nc,ne);

  sing_id_list.resize(nb*nc);
  sing_t_list.resize(nb*nc);

  for(int i=0; i<nc; i++){
    for(int j=0; j<ne; j++){
      if(TT(i,j)<=1 && TT(i,j)>=0){
        sing_id_list[i].emplace_back(j);
        sing_t_list[i].emplace_back(TT(i,j));
      }
    }
  }
  for(int k=1; k<nb; k++){
    for(int i=0; i<nc; i++){
      for(int j=0; j<sing_id_list[i].size(); j++){
        sing_id_list[k*nc+i].emplace_back(sing_id_list[i][j]+k*ne);
        sing_t_list[k*nc+i].emplace_back(sing_t_list[i][j]);
      }
    }
  }
}


void infinite::singular_on_bnd(const Eigen::VectorXd &xe, 
                     const Eigen::VectorXd &xc,
                     const int &nb,
                     std::vector<std::vector<int>> &sing_edge_list,
                     std::vector<std::vector<int>> &sing_end_list,
                     std::vector<std::vector<double>> &edge_t_list)
{
  int nc = xc.size();   //number of sample point
  int ne = xe.size()-1; //number of line segment
  // double th = (double)ne*0.005;
  double th = 0.005;
  double uv = 1.0+th;
  double lv = -th;
  Eigen::MatrixXd TT;

  TT.resize(nc,ne);
  for(int j=0; j<nc; j++){
    TT.row(j) = (xc(j)-xe.head(ne).array())/
      (xe.tail(ne).array()-xe.head(ne).array());
  }
  sing_edge_list.resize(nb*nc);
  sing_end_list.resize(nb*nc);
  edge_t_list.resize(nb*nc);

  for(int i=0; i<nc; i++){
    for(int j=0; j<ne; j++){
      if(TT(i,j)<uv && TT(i,j)>lv){
        sing_edge_list[i].emplace_back(j);
        edge_t_list[i].emplace_back(TT(i,j));
        if(abs(TT(i,j))<0.01*th ||abs(1-TT(i,j))<0.01*th){
          sing_end_list[i].emplace_back(j);
        }
      }
    }
  }
  for(int k=1; k<nb; k++){
    for(int i=0; i<nc; i++){
      for(int j=0; j<sing_edge_list[i].size(); j++){
        sing_edge_list[k*nc+i].emplace_back(sing_edge_list[i][j]+k*ne);
        edge_t_list[k*nc+i].emplace_back(edge_t_list[i][j]);
      }
      for(int j=0; j<sing_end_list[i].size(); j++){
        sing_end_list[k*nc+i].emplace_back(sing_end_list[i][j]+k*ne);
      }    
    }
  }
}

void infinite::singular_on_bnd(const Eigen::VectorXd &xe, 
                     const Eigen::VectorXd &xc,
                     const Eigen::VectorXd &S,
                     Eigen::MatrixXd &G,
                     Eigen::MatrixXd &F)
{
  int nc = xc.size();   //number of sample point
  int ne = xe.size()-1; //number of line segment
  int nb = S.size()/ne; //number of curve
  double th = (double)ne*0.005;
  double uv = 1.0+th;
  double lv = -th;
  Eigen::MatrixXd TT;

  TT.resize(nc,ne);
  for(int j=0; j<nc; j++){
    TT.row(j) = (xc(j)-xe.head(ne).array())/
      (xe.tail(ne).array()-xe.head(ne).array());
  }

  std::vector<int> edge_i_list, edge_j_list,
                   end_i_list, end_j_list;
  std::vector<double> edge_t_list;
  for(int i=0; i<TT.rows(); i++){
    for(int j=0; j<TT.cols(); j++){
      if(TT(i,j)<uv && TT(i,j)>lv){
        edge_i_list.emplace_back(i);
        edge_j_list.emplace_back(j);
        edge_t_list.emplace_back(TT(i,j));
        // if(abs(TT(i,j))<1e-4 ||abs(1-TT(i,j))<1e-4){
        if(abs(TT(i,j))<0.01*th ||abs(1-TT(i,j))<0.01*th){
          end_i_list.emplace_back(i);
          end_j_list.emplace_back(j);
        }
      }
    }
  }
  

  Eigen::VectorXi edgeI = Eigen::Map<Eigen::VectorXi>(edge_i_list.data(),edge_i_list.size());
  Eigen::VectorXi edgeJ = Eigen::Map<Eigen::VectorXi>(edge_j_list.data(),edge_j_list.size());
  Eigen::VectorXd edgeT = Eigen::Map<Eigen::VectorXd>(edge_t_list.data(),edge_t_list.size());

  Eigen::VectorXi edgeII(edgeI.size()*nb);
  Eigen::VectorXi edgeJJ(edgeJ.size()*nb);

  for(int k=0; k<nb; k++){
    edgeII.segment(k*edgeI.size(),edgeI.size()) = edgeI.array()+k*nc;
    edgeJJ.segment(k*edgeJ.size(),edgeJ.size()) = edgeJ.array()+k*ne;
  }

  Eigen::VectorXd s_edge = igl::slice(S,edgeJJ);
  Eigen::VectorXd toe = s_edge.array()*edgeT.replicate(nb,1).array();

  for(int k=0;k<edgeII.size(); k++)
  {
    G(edgeII(k),edgeJJ(k)) = ((toe(k)-s_edge(k))*log((toe(k)-s_edge(k))*(toe(k)-s_edge(k)))
                              -toe(k)*log(toe(k)*toe(k))+2*s_edge(k))/(4*igl::PI);
    F(edgeII(k),edgeJJ(k)) = 0;
  }
  if(end_i_list.size()>0){
    Eigen::VectorXi endI = Eigen::Map<Eigen::VectorXi>(end_i_list.data(),end_i_list.size());
    Eigen::VectorXi endJ = Eigen::Map<Eigen::VectorXi>(end_j_list.data(),end_j_list.size());
    Eigen::VectorXi endII(endI.size()*nb);
    Eigen::VectorXi endJJ(endJ.size()*nb);
    for(int k=0; k<nb; k++){
      endII.segment(k*endI.size(),endI.size()) = endI.array()+k*nc;
      endJJ.segment(k*endJ.size(),endJ.size()) = endJ.array()+k*ne;
    }
    Eigen::VectorXd s_end = igl::slice(S,endJJ);
    for(int k=0;k<endII.size(); k++){
      G(endII(k),endJJ(k)) = -(s_end(k)*log(s_end(k)*s_end(k))+2*s_end(k))/(4*igl::PI);
    }
  }
}
