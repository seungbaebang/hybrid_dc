#include "query_cell.h"
#include <iostream>

Eigen::RowVector2i infinite::get_node_descent(const int &index)
{
  Eigen::RowVector2i node_descent;
  if((index-1)%4==0)
    node_descent<<-1,-1;
  else if((index-1)%4==1)
    node_descent<<1,-1;
  else if((index-1)%4==2)
    node_descent<<-1,1;
  else if((index-1)%4==3)
    node_descent<<1,1;

  return node_descent;
}

Eigen::RowVector2i infinite::query_direction(int index, 
                        int neighbour_index, 
                        const Eigen::VectorXi &PA, 
                        const Eigen::VectorXi &LV)
{
  if(LV(index)!=LV(neighbour_index)){
    if(LV(index)>LV(neighbour_index)){
      while(true){
        index = PA(index);
        if(LV(index)==LV(neighbour_index))
          break;
      }
    }
    else{
      while(true){
        neighbour_index=PA(neighbour_index);
        if(LV(index)==LV(neighbour_index))
          break;
      }
    }
  }
  // Eigen::RowVector2i descent = get_node_descent(index);
  // Eigen::RowVector2i neighbour_descent = get_node_descent(neighbour_index);
  // Eigen::RowVector2i direction=Eigen::RowVector2i(descent.array()*neighbour_descent.array()-1)/2;
  Eigen::RowVector2i direction = get_node_descent(neighbour_index)-get_node_descent(index);
  // std::cout<<"direction 0 : "<<direction<<std::endl;
  if(PA(index)>0 && PA(neighbour_index)>0){
    // direction = -Eigen::RowVector2i(get_node_descent(PA(neighbour_index)).array()*direction.array());
    direction = (get_node_descent(PA(index)).array()*get_node_descent(PA(neighbour_index)).array()*direction.array())/2;
  }
  else{
    // direction = -Eigen::RowVector2i(neighbour_descent.array()*direction.array());
    direction = direction/2;
  }
  return direction;
}

void infinite::update_cells(const Eigen::MatrixX4i &CH, 
                    const Eigen::VectorXi &PA,
                    const Eigen::MatrixXd &CN,
                    const Eigen::VectorXd &W,
                    const Eigen::Matrix2d &pp,
                    const int &pid,
                    std::vector<std::vector<int> > &P_EI,
                    RowVec2d_list_list &P_LL,
                    Mat2d_list_list &P_PP,
                    std::vector<int> qEI_list)
{
  std::function<void(int &)> adap_cell;
  adap_cell = [&adap_cell,&CH,&CN,&W,&pp,&P_EI,&P_LL,&P_PP,
              &pid,&qEI_list]//,&pid_list,&rns_list,&nps_list]
  (int &index)->void
  {
    Eigen::RowVector2d cn = CN.row(index);
    double hw = W(index)/2;
    Eigen::Matrix2d bnd;
    bnd(0,0) = cn(0)-hw;
    bnd(1,0) = cn(0)+hw;
    bnd(0,1) = cn(1)-hw;
    bnd(1,1) = cn(1)+hw;

    Eigen::RowVector2d np1, np2, rns; 
    if(liang_barsky_clipper(bnd,pp.row(0),pp.row(1),np1,np2,rns)){
      qEI_list.emplace_back(index);
      // pid_list.emplace_back(pid);
      // rns_list.emplace_back(rns);
      Eigen::Matrix2d nps;
      nps.row(0) = np1;
      nps.row(1) = np2;
      // nps_list.emplace_back(nps);

      P_EI.at(index).emplace_back(pid);
      P_LL.at(index).emplace_back(rns);
      P_PP.at(index).emplace_back(nps);
      // break;
    }
    if(CH(index,0)==-1){
      return;
    }
    else{
      for(int ci=0; ci<4; ci++){
        int child_index = CH(index,ci);
        adap_cell(child_index);
      }
    }
  };
  int cell_id=0;
  adap_cell(cell_id);
}

void infinite::update_cells(const Eigen::MatrixX4i &CH, 
                    const Eigen::VectorXi &PA,
                    const Eigen::MatrixXd &CN,
                    const Eigen::VectorXd &W,
                    const Eigen::Matrix2d &pp,
                    const int &pid,
                    std::vector<std::vector<int> > &P_EI,
                    RowVec2d_list_list &P_LL,
                    Mat2d_list_list &P_PP)
{
  std::vector<int> qEI_list;
  // std::vector<int> pid_list;
  // RowVec2d_list rns_list;
  // Mat2d_list nps_list;

  std::function<void(int &)> adap_cell;
  adap_cell = [&adap_cell,&CH,&CN,&W,&pp,&P_EI,&P_LL,&P_PP,
              &pid,&qEI_list]//,&pid_list,&rns_list,&nps_list]
  (int &index)->void
  {
    Eigen::RowVector2d cn = CN.row(index);
    double hw = W(index)/2;
    Eigen::Matrix2d bnd;
    bnd(0,0) = cn(0)-hw;
    bnd(1,0) = cn(0)+hw;
    bnd(0,1) = cn(1)-hw;
    bnd(1,1) = cn(1)+hw;

    Eigen::RowVector2d np1, np2, rns; 
    if(liang_barsky_clipper(bnd,pp.row(0),pp.row(1),np1,np2,rns)){
      qEI_list.emplace_back(index);
      // pid_list.emplace_back(pid);
      // rns_list.emplace_back(rns);
      Eigen::Matrix2d nps;
      nps.row(0) = np1;
      nps.row(1) = np2;
      // nps_list.emplace_back(nps);

      P_EI.at(index).emplace_back(pid);
      P_LL.at(index).emplace_back(rns);
      P_PP.at(index).emplace_back(nps);
      // break;
    }
    if(CH(index,0)==-1){
      return;
    }
    else{
      for(int ci=0; ci<4; ci++){
        int child_index = CH(index,ci);
        adap_cell(child_index);
      }
    }
  };
  int cell_id=0;
  adap_cell(cell_id);

  // for(int i=0; i<qEI_list.size(); i++){
  //   int cid = qEI_list[i];
  //   if(PA(cid)<=0)
  //     continue;
  //   cid=PA(cid);
  //   while(true){
  //     if(std::find(P_EI[cid].begin(),P_EI[cid].end(),pid)==P_EI[cid].end())
  //       P_EI[cid].emplace_back(pid);
  //     else
  //       break;//sure?
  //     if(PA(cid)<=0)
  //       break;
  //     cid=PA(cid);
  //   }
  // }
}


int infinite::query_cell(const Eigen::MatrixX4i &CH, 
               const Eigen::MatrixXd &CN,
               const Eigen::VectorXd &W,
               const Eigen::RowVector2d &q)
{
  std::function<void(int &)> find_min_dist_cell;
  find_min_dist_cell = [&find_min_dist_cell,&CH,&CN,&W,&q]
  (int &index)->void
  {
    if(CH(index,0)==-1){
      return;
    }
    else{
      for(int ci=0; ci<4; ci++){
        int child_index = CH(index,ci);
        Eigen::RowVector2d cn = CN.row(child_index);
        double hw = W(child_index)/2;
        double ux=cn(0)+hw; double lx=cn(0)-hw;
        double uy=cn(1)+hw; double ly=cn(1)-hw;
        if(q(0)<=ux && q(0)>=lx && q(1)<=uy && q(1)>=ly){
          index=child_index;
          break;
        }
      }
      find_min_dist_cell(index);
    }
  };
  int cell_id=0;
  find_min_dist_cell(cell_id);
  return cell_id;
}


int infinite::query_cell(const Eigen::MatrixX4i &CH, 
               const Eigen::VectorXcd &CNi,
               const std::complex<double> &q)
{
  std::function<void(int &)> find_min_dist_cell;
  find_min_dist_cell = [&find_min_dist_cell,&CH,&CNi,&q]
  (int &index)->void
  {
    if(CH(index,0)==-1){
      return;
    }
    else{
      double min_dist = std::numeric_limits<double>::max();
      int child_id = -1;
      for(int ci=0; ci<4; ci++){
        int child_index = CH(index,ci);
        std::complex<double> cn = CNi(child_index);
        double dist = std::abs(q-cn);
        if(dist<min_dist){
          min_dist = dist;
          child_id = child_index;
        }
      }
      index = child_id;
      //std::cout<<"index: "<<index<<std::endl;
      find_min_dist_cell(index);
    }
  };
  int cell_id=0;
  find_min_dist_cell(cell_id);
  return cell_id;
}