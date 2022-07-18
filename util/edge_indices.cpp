#include "edge_indices.h"

Eigen::MatrixX2i edge_indices(int nb, int ne)
{
  Eigen::MatrixX2i E(nb*ne,2);
  for(int k=0; k<nb; k++){
    E.block(k*ne,0,ne,1)=Eigen::VectorXi::LinSpaced(ne,0,ne-1).array()+k*(ne+1);
    E.block(k*ne,1,ne,1)=Eigen::VectorXi::LinSpaced(ne,1,ne).array()+k*(ne+1);
  }
  return E;
}

void edge_indices(int nb, int ne, MatX2i &E)
{
  E.resize(nb*ne,2);
  for(int k=0; k<nb; k++){
    E.block(k*ne,0,ne,1)=Eigen::VectorXi::LinSpaced(ne,0,ne-1).array()+k*(ne+1);
    E.block(k*ne,1,ne,1)=Eigen::VectorXi::LinSpaced(ne,1,ne).array()+k*(ne+1);
  }
}

void edge_indices(const Eigen::VectorXi &nel, Eigen::MatrixX2i &E)
{
  int np=0;
  for(int k=0; k<nel.size(); k++){
    np=np+nel(k);
  }
  E.resize(np,2);
  np=0;
  for(int k=0; k<nel.size(); k++){
    int ne = nel(k);
    E.block(np,0,ne,1)=Eigen::VectorXi::LinSpaced(ne,0,ne-1).array()+np+k;
    E.block(np,1,ne,1)=Eigen::VectorXi::LinSpaced(ne,1,ne).array()+np+k;
    np=np+ne;
  }
}

// template <class Mat>
// Mat edge_indices(int nb, int ne)
// {
//   Mat E(nb*ne,2);
//   for(int k=0; k<nb; k++){
//     E.block(k*ne,0,ne,1)=Eigen::VectorXi::LinSpaced(ne,0,ne-1).array()+k*(ne+1);
//     E.block(k*ne,1,ne,1)=Eigen::VectorXi::LinSpaced(ne,1,ne).array()+k*(ne+1);
//   }
//   return E;
// }