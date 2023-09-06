#include "quadtree.h"

#include <iostream>
#include <igl/get_seconds.h>

Eigen::RowVector2d translate_center(const Eigen::RowVector2d &parent_center, 
                      const double h,
                      const int child_index)
{
  Eigen::RowVector2d change_vector;
  change_vector << -h, -h;

  //positive x chilren are 1,3
  if (child_index % 2) {
    change_vector(0) = h;
  }
  //positive y children are 2,3
  if (child_index == 2 || child_index == 3) {
    change_vector(1) = h;
  }
  Eigen::RowVector2d output = parent_center + change_vector;
  return output;
}

int get_quad(const Eigen::RowVector2d &location,
             const Eigen::RowVector2d &center)
{
  int index = 0;
  if (location(0) >= center(0)) {
    index = index + 1;
  }
  if (location(1) >= center(1)) {
    index = index + 2;
  }
  return index;
}

void infinite::determine_update_cells(
            const std::vector<int> &pre_leaf_cells,
            const std::vector<int> &cur_leaf_cells,
            const Eigen::VectorXi &new_cells,
            const Eigen::VectorXi &new_pids,
            const std::vector<std::vector<int > > &pre_Q_PEI,
            const std::vector<std::vector<int > > &cur_Q_PEI,
            std::vector<int> &update_leafs)
{


  std::set<int> new_pid_set;
  for(int i=0; i<new_pids.size(); i++){
    new_pid_set.insert(new_pids(i));
  }
  std::set<int> new_cell_set;
  for(int i=0; i<new_cells.size(); i++){
    new_cell_set.insert(new_cells(i));
  }
  
  update_leafs.clear();
  for(int i=0; i<cur_leaf_cells.size(); i++){
    int li = cur_leaf_cells[i];
    if(new_cell_set.find(li)!=new_cell_set.end()){
      update_leafs.emplace_back(li);
      continue;
    }
    if(cur_Q_PEI[li].size()!=pre_Q_PEI[li].size()){
      update_leafs.emplace_back(li);
      continue;
    }
    for(int j=0; j<cur_Q_PEI[li].size(); j++){
      if(new_pid_set.find(cur_Q_PEI[li][j])!=new_pid_set.end()){
        update_leafs.emplace_back(li);
        break;
      }
    }
  }

}

void infinite::determine_update_cells(
            const std::vector<int> &pre_leaf_cells,
            const std::vector<int> &cur_leaf_cells,
            const Eigen::VectorXi &new_cells,
            const Eigen::VectorXi &new_pids,
            const std::vector<std::vector<int > > &pre_Q_PEI,
            const std::vector<std::vector<int > > &cur_Q_PEI,
            const std::vector<std::vector<int> > &big_seps,
            const std::vector<int> &big_sep_updates,
            std::vector<int> &update_leafs,
            std::vector<int> &update_big_seps)
{
  std::set<int> new_pid_set;
  for(int i=0; i<new_pids.size(); i++){
    new_pid_set.insert(new_pids(i));
  }
  std::set<int> new_cell_set;
  for(int i=0; i<new_cells.size(); i++){
    new_cell_set.insert(new_cells(i));
  }
  
  update_leafs.clear();
  for(int i=0; i<cur_leaf_cells.size(); i++){
    int li = cur_leaf_cells[i];
    if(new_cell_set.find(li)!=new_cell_set.end()){
      update_leafs.emplace_back(li);
      continue;
    }
    if(cur_Q_PEI[li].size()!=pre_Q_PEI[li].size()){
      update_leafs.emplace_back(li);
      continue;
    }
    for(int j=0; j<cur_Q_PEI[li].size(); j++){
      if(new_pid_set.find(cur_Q_PEI[li][j])!=new_pid_set.end()){
        update_leafs.emplace_back(li);
        break;
      }
    }
  }

  std::set<int> update_leaf_set;
  for(int i=0; i<update_leafs.size(); i++){
    update_leaf_set.insert(update_leafs[i]);
  }

  update_big_seps.clear();
  update_big_seps = big_sep_updates;
  for(int i=0; i<big_seps.size(); i++){
    if(std::find(update_big_seps.begin(),update_big_seps.end(),i)!=update_big_seps.end())
      continue;
    if(update_leaf_set.find(i)!=update_leaf_set.end()){
      update_big_seps.emplace_back(i);
      continue;
    }
    for(int j=0; j<big_seps[i].size(); j++){
      if(update_leaf_set.find(big_seps[i][j])!=update_leaf_set.end()){
        update_big_seps.emplace_back(i);
        break;
      }
    }
  }
}




void infinite::adap_quadtree(const Eigen::VectorXi &cI,
            const std::vector<std::vector<int> > &cEI,
            const Eigen::VectorXi &qI,
            const Eigen::VectorXi &sub_pids,
            const Eigen::VectorXi &sub_qids,
            const Eigen::VectorXi &new_pids,
            const Eigen::VectorXi &new_qids,
            const Eigen::MatrixXd &nP,
            const Eigen::MatrixXi &nE,
            const Eigen::MatrixXd &nC,
            const Eigen::MatrixXd &nQ,
            const int& max_depth,
            const int& min_pnt_num,
            std::vector<std::vector<int> > &P_I,
            std::vector<std::vector<int> > &P_EI,
            std::vector<std::vector<int> > &Q_I,
            RowVec2d_list_list &P_LL,
            Mat2d_list_list &P_PP,
            Eigen::MatrixXi &CH,
            Eigen::VectorXi &PA,
            Eigen::VectorXi &LV,
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W,
            Eigen::VectorXi &SC)
{
  //I_cells, EI_cells, QI_cells can be removed
  std::vector<int> I_cells, EI_cells;
  for(int i=0; i<sub_pids.size(); i++){
    int sid = sub_pids(i);
    int cid = cI(sid);
    while(true){
      P_I[cid].erase(std::remove(P_I[cid].begin(),P_I[cid].end(),sid),P_I[cid].end());
      if(std::find(I_cells.begin(),I_cells.end(),cid)==I_cells.end())
        I_cells.emplace_back(cid);
      if(PA(cid)<0)
        break;
      cid=PA(cid);
    }
    std::vector<int> cid_list = cEI[sid];
    for(int j=0; j<cid_list.size(); j++){
      int cid = cid_list[j];
      while(true){
        auto it = std::find(P_EI[cid].begin(),P_EI[cid].end(),sid);
        if(it!=P_EI[cid].end()){
          int del_id = it-P_EI[cid].begin();
          P_EI[cid].erase(P_EI[cid].begin()+del_id);
          P_LL[cid].erase(P_LL[cid].begin()+del_id);
          P_PP[cid].erase(P_PP[cid].begin()+del_id);
          if(std::find(EI_cells.begin(),EI_cells.end(),cid)==EI_cells.end())
            EI_cells.emplace_back(cid);
          if(PA(cid)<=0)
            break;
          cid=PA(cid);
        }
        else
          break;
      }
    }    
  }


  std::vector<int> QI_cells;
  for(int i=0; i<sub_qids.size(); i++){
    int sid = sub_qids(i);
    int cid = qI(sid);
    while(true){
      Q_I[cid].erase(std::remove(Q_I[cid].begin(),Q_I[cid].end(),sid),Q_I[cid].end());
      if(std::find(QI_cells.begin(),QI_cells.end(),cid)==QI_cells.end())
        QI_cells.emplace_back(cid);
      if(PA(cid)<0)
        break;
      cid=PA(cid);
    }  
  }


  // for(int ci=0; ci<I_cells.size(); ci++){
  for(int ci=0; ci<P_I.size(); ci++){
    int cid = ci;//I_cells[ci];
    for(int j=0; j<P_I[cid].size(); j++){
      int pid = P_I[cid][j];
      int val_to_sub=0;
      for(int i=0; i<sub_pids.size(); i++){
        if(pid>sub_pids[i])
          val_to_sub++;
      }
      if(val_to_sub==0)
        continue;
      pid=pid-val_to_sub;
      P_I[cid][j]=pid;
    }
  }

  // for(int ci=0; ci<EI_cells.size(); ci++){
  for(int ci=0; ci<P_EI.size(); ci++){
    int cid = ci;//EI_cells[ci];
    for(int j=0; j<P_EI[cid].size(); j++){
      int pid = P_EI[cid][j];
      int val_to_sub=0;
      for(int i=0; i<sub_pids.size(); i++){
        if(pid>sub_pids[i])
          val_to_sub++;
      }
      if(val_to_sub==0)
        continue;
      pid=pid-val_to_sub;
      P_EI[cid][j]=pid;
    }
  }

  // for(int ci=0; ci<QI_cells.size(); ci++){
  for(int ci=0; ci<Q_I.size(); ci++){
    int cid = ci;//QI_cells[ci];
    for(int j=0; j<Q_I[cid].size(); j++){
      int qid = Q_I[cid][j];
      int val_to_sub=0;
      for(int i=0; i<sub_qids.size(); i++){
        if(qid>sub_qids[i])
          val_to_sub++;
      }
      if(val_to_sub==0)
        continue;
      qid=qid-val_to_sub;
      Q_I[cid][j]=qid;
    }
  }


  std::vector<int> sub_cells;
  for(int i=0; i<new_pids.size(); i++){
    int cid = query_cell(CH,CN,W,nC.row(new_pids(i)));
    sub_cells.emplace_back(cid);
    while(true){
      if(std::find(P_I[cid].begin(),P_I[cid].end(),new_pids[i])==P_I[cid].end())
        P_I[cid].emplace_back(new_pids[i]);
      else
        break;//sure?
      if(PA(cid)<=0)
        break;
      cid=PA(cid);
    }
    Eigen::Matrix2d pp;
    pp.row(0) = nP.row(nE(new_pids(i),0));
    pp.row(1) = nP.row(nE(new_pids(i),1));
    update_cells(CH,PA,CN,W,pp,new_pids[i],P_EI,P_LL,P_PP);
  }


  for(int i=0; i<new_qids.size(); i++){
    int cid = query_cell(CH,CN,W,nQ.row(new_qids(i)));
    while(true){
      if(std::find(Q_I[cid].begin(),Q_I[cid].end(),new_qids[i])==Q_I[cid].end())
        Q_I[cid].emplace_back(new_qids[i]);
      else
        break;//sure?
      if(PA(cid)<=0)
        break;
      cid=PA(cid);
    }
  }


  std::vector<int> u_sub_cells;
  igl::unique(sub_cells,u_sub_cells);
  sub_cells=u_sub_cells;

  std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children(CH.rows());
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers(CN.rows());

  int m = 0;
  std::vector<int> parent(PA.size());
  std::vector<int> levels(LV.size());
  std::vector<double> widths(W.size());

  for(int i=0; i<CH.rows(); i++){
    children[i]=CH.row(i);
  }
  for(int i=0; i<CN.rows(); i++){
    centers[i]=CN.row(i);
  }
  for(int i=0; i<PA.size(); i++){
    parent[i]=PA(i);
  }
  for(int i=0; i<LV.size(); i++){
    levels[i]=LV(i);
  }
  for(int i=0; i<W.size(); i++){
    widths[i]=W(i);
  }
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int)> update_quadtree;
  update_quadtree = [&update_quadtree,&m,&zero_to_three,&neg_ones,
          &nP,&nE,&nC,&nQ,&P_I,&P_EI,&Q_I,&P_LL,&P_PP,&children,&parent,&levels,
          &centers,&widths,&min_pnt_num,&max_depth]
  (const int index) -> void{
    if(P_I.at(index).size()>min_pnt_num && levels.at(index)<max_depth){

      children.at(index)=zero_to_three.array() + m;
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);
      int depth = levels.at(index)+1;


      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
        Q_I.emplace_back(std::vector<int>());
        P_EI.emplace_back(std::vector<int>());
        P_LL.emplace_back(RowVec2d_list());
        P_PP.emplace_back(Mat2d_list());
      }
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(nC.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(nQ.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int i = 0; i < 4; i++) {
        Eigen::Matrix2d bnd;
        bnd(0,0) = centers[m+i](0)-h/2;
        bnd(1,0) = centers[m+i](0)+h/2;
        bnd(0,1) = centers[m+i](1)-h/2;
        bnd(1,1) = centers[m+i](1)+h/2;

        for (int j = 0; j < P_EI.at(index).size(); j++) {
          int curr_point_index = P_EI.at(index).at(j);
          Eigen::RowVector2d p1 = nP.row(nE(curr_point_index,0));
          Eigen::RowVector2d p2 = nP.row(nE(curr_point_index,1));

          Eigen::RowVector2d np1, np2, rns; 
          if(liang_barsky_clipper(bnd,p1,p2,np1,np2,rns)){
            P_EI.at(m+i).emplace_back(curr_point_index);
            P_LL.at(m+i).emplace_back(rns);
            Eigen::Matrix2d nps;
            nps.row(0) = np1;
            nps.row(1) = np2;
            P_PP.at(m+i).emplace_back(nps);
          }
        }
      }
      m += 4;
      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        update_quadtree(children.at(index)(i));
      }
    }
  };

  for(int i=0; i<sub_cells.size(); i++){
    int cid=sub_cells[i];
    if(P_I[cid].size()>min_pnt_num && LV(cid)<max_depth){
      m=P_I.size();
      update_quadtree(cid);
    }
  }

  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }

  SC.resize(sub_cells.size(),1);
  for (int i = 0; i < sub_cells.size(); i++) {
    SC(i) = sub_cells.at(i);
  }
}



void infinite::adap_quadtree(const Eigen::VectorXi &cI,
            const std::vector<std::vector<int> > &cEI,
            const Eigen::VectorXi &sub_ids,
            const Eigen::MatrixXd &nP,
            const Eigen::MatrixXi &nE,
            const Eigen::MatrixXd &nC,
            const Eigen::MatrixXd &Q,
            const Eigen::VectorXi &new_ids,
            const int& max_depth,
            const int& min_pnt_num,
            std::vector<std::vector<int> > &P_I,
            std::vector<std::vector<int> > &P_EI,
            std::vector<std::vector<int> > &Q_I,
            RowVec2d_list_list &P_LL,
            Mat2d_list_list &P_PP,
            Eigen::MatrixXi &CH,
            Eigen::VectorXi &PA,
            Eigen::VectorXi &LV,
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W,
            Eigen::VectorXi &SC)
{
  std::vector<int> I_cells, EI_cells;
  for(int i=0; i<sub_ids.size(); i++){
    int sid = sub_ids(i);
    int cid = cI(sid);
    while(true){
      P_I[cid].erase(std::remove(P_I[cid].begin(),P_I[cid].end(),sid),P_I[cid].end());
      if(std::find(I_cells.begin(),I_cells.end(),cid)==I_cells.end())
        I_cells.emplace_back(cid);
      if(PA(cid)<0)
        break;
      cid=PA(cid);
    }
    std::vector<int> cid_list = cEI[sid];
    for(int j=0; j<cid_list.size(); j++){
      int cid = cid_list[j];
      while(true){
        auto it = std::find(P_EI[cid].begin(),P_EI[cid].end(),sid);
        if(it!=P_EI[cid].end()){
          int del_id = it-P_EI[cid].begin();
          P_EI[cid].erase(P_EI[cid].begin()+del_id);
          P_LL[cid].erase(P_LL[cid].begin()+del_id);
          P_PP[cid].erase(P_PP[cid].begin()+del_id);
          if(std::find(EI_cells.begin(),EI_cells.end(),cid)==EI_cells.end())
            EI_cells.emplace_back(cid);
          if(PA(cid)<=0)
            break;
          cid=PA(cid);
        }
        else
          break;
      }
    }    
  }

  for(int ci=0; ci<I_cells.size(); ci++){
    int cid = I_cells[ci];
    for(int j=0; j<P_I[cid].size(); j++){
      int pid = P_I[cid][j];
      int val_to_sub=0;
      for(int i=0; i<sub_ids.size(); i++){
        if(pid>sub_ids[i])
          val_to_sub++;
      }
      if(val_to_sub==0)
        continue;
      pid=pid-val_to_sub;
      P_I[cid][j]=pid;
    }
  }

  for(int ci=0; ci<EI_cells.size(); ci++){
    int cid = EI_cells[ci];
    for(int j=0; j<P_EI[cid].size(); j++){
      int pid = P_EI[cid][j];
      int val_to_sub=0;
      for(int i=0; i<sub_ids.size(); i++){
        if(pid>sub_ids[i])
          val_to_sub++;
      }
      if(val_to_sub==0)
        continue;
      pid=pid-val_to_sub;
      P_EI[cid][j]=pid;
    }
  }

  std::vector<int> sub_cells;
  for(int i=0; i<new_ids.size(); i++){
    int cid = query_cell(CH,CN,W,nC.row(new_ids(i)));
    sub_cells.emplace_back(cid);
    while(true){
      if(std::find(P_I[cid].begin(),P_I[cid].end(),new_ids[i])==P_I[cid].end())
        P_I[cid].emplace_back(new_ids[i]);
      else
        break;//sure?
      if(PA(cid)<=0)
        break;
      cid=PA(cid);
    }
    Eigen::Matrix2d pp;
    pp.row(0) = nP.row(nE(new_ids(i),0));
    pp.row(1) = nP.row(nE(new_ids(i),1));
    update_cells(CH,PA,CN,W,pp,new_ids[i],P_EI,P_LL,P_PP);
  }

  std::vector<int> u_sub_cells;
  igl::unique(sub_cells,u_sub_cells);
  sub_cells=u_sub_cells;

  std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children(CH.rows());
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers(CN.rows());

  int m = 0;
  std::vector<int> parent(PA.size());
  std::vector<int> levels(LV.size());
  std::vector<double> widths(W.size());

  for(int i=0; i<CH.rows(); i++){
    children[i]=CH.row(i);
  }
  for(int i=0; i<CN.rows(); i++){
    centers[i]=CN.row(i);
  }
  for(int i=0; i<PA.size(); i++){
    parent[i]=PA(i);
  }
  for(int i=0; i<LV.size(); i++){
    levels[i]=LV(i);
  }
  for(int i=0; i<W.size(); i++){
    widths[i]=W(i);
  }
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int)> update_quadtree;
  update_quadtree = [&update_quadtree,&m,&zero_to_three,&neg_ones,
          &nP,&nE,&nC,&Q,&P_I,&P_EI,&Q_I,&P_LL,&P_PP,&children,&parent,&levels,
          &centers,&widths,&min_pnt_num,&max_depth]
  (const int index) -> void{
    if(P_I.at(index).size()>min_pnt_num && levels.at(index)<max_depth){

      children.at(index)=zero_to_three.array() + m;
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);
      int depth = levels.at(index)+1;


      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
        Q_I.emplace_back(std::vector<int>());
        P_EI.emplace_back(std::vector<int>());
        P_LL.emplace_back(RowVec2d_list());
        P_PP.emplace_back(Mat2d_list());
      }
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(nC.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(Q.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int i = 0; i < 4; i++) {
        Eigen::Matrix2d bnd;
        bnd(0,0) = centers[m+i](0)-h/2;
        bnd(1,0) = centers[m+i](0)+h/2;
        bnd(0,1) = centers[m+i](1)-h/2;
        bnd(1,1) = centers[m+i](1)+h/2;

        for (int j = 0; j < P_EI.at(index).size(); j++) {
          int curr_point_index = P_EI.at(index).at(j);
          Eigen::RowVector2d p1 = nP.row(nE(curr_point_index,0));
          Eigen::RowVector2d p2 = nP.row(nE(curr_point_index,1));

          Eigen::RowVector2d np1, np2, rns; 
          if(liang_barsky_clipper(bnd,p1,p2,np1,np2,rns)){
            P_EI.at(m+i).emplace_back(curr_point_index);
            P_LL.at(m+i).emplace_back(rns);
            Eigen::Matrix2d nps;
            nps.row(0) = np1;
            nps.row(1) = np2;
            P_PP.at(m+i).emplace_back(nps);
          }
        }
      }
      m += 4;
      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        update_quadtree(children.at(index)(i));
      }
    }
  };

  for(int i=0; i<sub_cells.size(); i++){
    int cid=sub_cells[i];
    if(P_I[cid].size()>min_pnt_num && LV(cid)<max_depth){
      m=P_I.size();
      update_quadtree(cid);
    }
  }

  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }

  SC.resize(sub_cells.size(),1);
  for (int i = 0; i < sub_cells.size(); i++) {
    SC(i) = sub_cells.at(i);
  }
}



void infinite::adap_quadtree(const Eigen::VectorXi &cI,
            const Eigen::VectorXi &sub_ids,
            const Eigen::MatrixXd &nC,
            const Eigen::VectorXi &new_ids,
            const int& max_depth,
            const int& min_pnt_num,
            std::vector<std::vector<int> > &P_I,
            Eigen::MatrixXi &CH,
            Eigen::VectorXi &PA,
            Eigen::VectorXi &LV,
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W,
            Eigen::VectorXi &SC)
{
  std::vector<int> cell_list;
  for(int i=0; i<sub_ids.size(); i++){
    int sid = sub_ids(i);
    int cid = cI(sid);
    while(true){
      P_I[cid].erase(std::remove(P_I[cid].begin(),P_I[cid].end(),sid),P_I[cid].end());
      cell_list.emplace_back(cid);
      if(PA(cid)<0)
        break;
      cid=PA(cid);
    }
  }
  std::vector<int> u_cell_list;
  igl::unique(cell_list,u_cell_list);
  for(int ci=0; ci<u_cell_list.size(); ci++){
    int cid = u_cell_list[ci];
    for(int j=0; j<P_I[cid].size(); j++){
      int pid = P_I[cid][j];
      int val_to_sub=0;
      for(int i=0; i<sub_ids.size(); i++){
        if(pid>sub_ids[i])
          val_to_sub++;
      }
      if(val_to_sub==0)
        continue;
      pid=pid-val_to_sub;
      P_I[cid][j]=pid;
    }
  }

  std::vector<int> sub_cells;
  for(int i=0; i<new_ids.size(); i++){
    int cid = query_cell(CH,CN,W,nC.row(new_ids(i)));
    sub_cells.emplace_back(cid);
    while(true){
      if(std::find(P_I[cid].begin(),P_I[cid].end(),new_ids[i])==P_I[cid].end())
        P_I[cid].emplace_back(new_ids[i]);
      else
        break;//sure?
      if(PA(cid)<0)
        break;
      cid=PA(cid);
    }
  }

  auto get_quad = [](const Eigen::RowVector2d &location,
    const Eigen::RowVector2d &center) {
    int index = 0;
    if (location(0) >= center(0)) {
      index = index + 1;
    }
    if (location(1) >= center(1)) {
      index = index + 2;
    }
    return index;
  };

  std::function<
      Eigen::RowVector2d(const Eigen::RowVector2d, const double,
          const int)> translate_center = [](
      const Eigen::RowVector2d &parent_center, const double h,
      const int child_index) {
    Eigen::RowVector2d change_vector;
    change_vector << -h, -h;

    //positive x chilren are 1,3
    if (child_index % 2) {
      change_vector(0) = h;
    }
    //positive y children are 2,3
    if (child_index == 2 || child_index == 3) {
      change_vector(1) = h;
    }
    Eigen::RowVector2d output = parent_center + change_vector;
    return output;
  };


  std::vector<int> u_sub_cells;
  igl::unique(sub_cells,u_sub_cells);
  sub_cells=u_sub_cells;

  std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children(CH.rows());
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers(CN.rows());

  int m = 0;
  std::vector<int> parent(PA.size());
  std::vector<int> levels(LV.size());
  std::vector<double> widths(W.size());

  for(int i=0; i<CH.rows(); i++){
    children[i]=CH.row(i);
  }
  for(int i=0; i<CN.rows(); i++){
    centers[i]=CN.row(i);
  }
  for(int i=0; i<PA.size(); i++){
    parent[i]=PA(i);
  }
  for(int i=0; i<LV.size(); i++){
    levels[i]=LV(i);
  }
  for(int i=0; i<W.size(); i++){
    widths[i]=W(i);
  }
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int)> helper;
  helper = [&helper,&translate_center,&get_quad,&m,&zero_to_three,&neg_ones,
          &nC,&P_I,&children,&parent,&levels,
          &centers,&widths,&min_pnt_num,&max_depth]
  (const int index) -> void{
    if(P_I.at(index).size()>min_pnt_num && levels.at(index)<max_depth){

      children.at(index)=zero_to_three.array() + m;
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);
      int depth = levels.at(index)+1;


      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
      }
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(nC.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      m += 4;
      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        helper(children.at(index)(i));
      }
    }
  };

  for(int i=0; i<sub_cells.size(); i++){
    int cid=sub_cells[i];
    if(P_I[cid].size()>min_pnt_num && LV(cid)<max_depth){
      m=P_I.size();
      helper(cid);
    }
  }
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);


  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }

  SC.resize(sub_cells.size(),1);
  for (int i = 0; i < sub_cells.size(); i++) {
    SC(i) = sub_cells.at(i);
  }
}


void infinite::quadtree_test(const Eigen::MatrixXd &P, 
             const Eigen::MatrixXi &E,
             const Eigen::MatrixXd &C, 
             const Eigen::MatrixXd &Q,
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& min_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &P_EI,
             std::vector<std::vector<int> > &Q_I,
             RowVec2d_list_list &P_LL,
             Mat2d_list_list &PP,
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::VectorXi &LV,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{
  std::cout<<"quadtree_test in"<<std::endl;
  double s = igl::get_seconds();
  std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  int m = 0;
  int opt_depth = 0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> build_quadtree;
  build_quadtree = [&build_quadtree,&m,&zero_to_three,&neg_ones,&P,&E,&C,&Q, 
      &P_I,&P_EI,&Q_I,&P_LL,&PP,&children,&parent,&levels, 
      &centers,&widths,&min_depth,&min_pnt_num,&opt_depth]
      (const int index, const int depth) -> void {
        // std::cout<<"P_I size: "<<P_I.at(index).size()<<", depth: "<<depth<<" min depth: "<<min_depth<<", test:"<<((P_I.at(index).size() > 1) || ((P_I.at(index).size()==1) && depth < min_depth))<<std::endl;
    if ((P_I.at(index).size() > 1) || 
        ((P_EI.at(index).size()>0) && depth < min_depth)) {

      children.at(index) = zero_to_three.array() + m;
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        Q_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
        P_EI.emplace_back(std::vector<int>());
        P_LL.emplace_back(RowVec2d_list());
        PP.emplace_back(Mat2d_list());
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(C.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(Q.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }

      for (int i = 0; i < 4; i++) {
        Eigen::Matrix2d bnd;
        bnd(0,0) = centers[m+i](0)-h/2;
        bnd(1,0) = centers[m+i](0)+h/2;
        bnd(0,1) = centers[m+i](1)-h/2;
        bnd(1,1) = centers[m+i](1)+h/2;

        for (int j = 0; j < P_EI.at(index).size(); j++) {
          int curr_point_index = P_EI.at(index).at(j);
          Eigen::RowVector2d p1 = P.row(E(curr_point_index,0));
          Eigen::RowVector2d p2 = P.row(E(curr_point_index,1));

          Eigen::RowVector2d np1, np2, rns; 
          if(liang_barsky_clipper(bnd,p1,p2,np1,np2,rns)){
            P_EI.at(m+i).emplace_back(curr_point_index);
            P_LL.at(m+i).emplace_back(rns);
            Eigen::Matrix2d nps;
            nps.row(0) = np1;
            nps.row(1) = np2;
            PP.at(m+i).emplace_back(nps);
          }
        }
      }
      //Now increase m
      m += 4;
      opt_depth = std::max(opt_depth,depth);
      for (int i = 0; i < 4; i++) {
        build_quadtree(children.at(index)(i), depth + 1);
      }
    }
  };

  {
    std::vector<int> allC(C.rows());
    for (int i = 0; i < allC.size(); i++)
      allC[i] = i;
    std::vector<int> allQ(Q.rows());
    for (int i = 0; i < allQ.size(); i++)
      allQ[i] = i;
    P_I.emplace_back(allC);
    P_EI.emplace_back(allC);    
    Q_I.emplace_back(allQ);
    P_LL.emplace_back(RowVec2d_list());
    PP.emplace_back(Mat2d_list());
  }

  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  build_quadtree(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }
  // LV.resize(opt_depth+1);
  // for (int i=0;i<levels.size(); i++){
  //   LV[levels[i]].emplace_back(i);
  // }
  double t = igl::get_seconds();
}




void infinite::quadtree(const Eigen::MatrixXd &P, 
             const Eigen::MatrixXi &E,
             const Eigen::MatrixXd &C, 
             const Eigen::MatrixXd &Q,
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& max_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &P_EI,
             std::vector<std::vector<int> > &Q_I,
             RowVec2d_list_list &P_LL,
             Mat2d_list_list &PP,
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::VectorXi &LV,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{
  double s = igl::get_seconds();
  std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  int m = 0;
  int opt_depth = 0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> build_quadtree;
  build_quadtree = [&build_quadtree,&m,&zero_to_three,&neg_ones,&P,&E,&C,&Q, 
      &P_I,&P_EI,&Q_I,&P_LL,&PP,&children,&parent,&levels, 
      &centers,&widths,&max_depth,&min_pnt_num,&opt_depth]
      (const int index, const int depth) -> void {
    if (P_I.at(index).size() > min_pnt_num && depth < max_depth) {

      children.at(index) = zero_to_three.array() + m;
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        Q_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
        P_EI.emplace_back(std::vector<int>());
        P_LL.emplace_back(RowVec2d_list());
        PP.emplace_back(Mat2d_list());
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(C.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(Q.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }

      for (int i = 0; i < 4; i++) {
        Eigen::Matrix2d bnd;
        bnd(0,0) = centers[m+i](0)-h/2;
        bnd(1,0) = centers[m+i](0)+h/2;
        bnd(0,1) = centers[m+i](1)-h/2;
        bnd(1,1) = centers[m+i](1)+h/2;

        for (int j = 0; j < P_EI.at(index).size(); j++) {
          int curr_point_index = P_EI.at(index).at(j);
          Eigen::RowVector2d p1 = P.row(E(curr_point_index,0));
          Eigen::RowVector2d p2 = P.row(E(curr_point_index,1));

          Eigen::RowVector2d np1, np2, rns; 
          if(liang_barsky_clipper(bnd,p1,p2,np1,np2,rns)){
            P_EI.at(m+i).emplace_back(curr_point_index);
            P_LL.at(m+i).emplace_back(rns);
            Eigen::Matrix2d nps;
            nps.row(0) = np1;
            nps.row(1) = np2;
            PP.at(m+i).emplace_back(nps);
          }
        }
      }
      //Now increase m
      m += 4;
      opt_depth = std::max(opt_depth,depth);
      for (int i = 0; i < 4; i++) {
        build_quadtree(children.at(index)(i), depth + 1);
      }
    }
  };

  {
    std::vector<int> allC(C.rows());
    for (int i = 0; i < allC.size(); i++)
      allC[i] = i;
    std::vector<int> allQ(Q.rows());
    for (int i = 0; i < allQ.size(); i++)
      allQ[i] = i;
    P_I.emplace_back(allC);
    P_EI.emplace_back(allC);    
    Q_I.emplace_back(allQ);
    P_LL.emplace_back(RowVec2d_list());
    PP.emplace_back(Mat2d_list());
  }

  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  build_quadtree(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }
  // LV.resize(opt_depth+1);
  // for (int i=0;i<levels.size(); i++){
  //   LV[levels[i]].emplace_back(i);
  // }
  double t = igl::get_seconds();
}



void infinite::quadtree(const Eigen::MatrixXd &P, 
             const Eigen::MatrixXi &E,
             const Eigen::MatrixXd &C, 
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& max_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &P_EI,
             RowVec2d_list_list &P_LL,
             Mat2d_list_list &cPE,
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::VectorXi &LV,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{
 std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  auto get_quad = [](const Eigen::RowVector2d &location,
    const Eigen::RowVector2d &center) {
    int index = 0;
    if (location(0) >= center(0)) {
      index = index + 1;
    }
    if (location(1) >= center(1)) {
      index = index + 2;
    }
    return index;
  };

  std::function<
      Eigen::RowVector2d(const Eigen::RowVector2d, const double,
          const int)> translate_center = [](
      const Eigen::RowVector2d &parent_center, const double h,
      const int child_index) {
    Eigen::RowVector2d change_vector;
    change_vector << -h, -h;

    //positive x chilren are 1,3
    if (child_index % 2) {
      change_vector(0) = h;
    }
    //positive y children are 2,3
    if (child_index == 2 || child_index == 3) {
      change_vector(1) = h;
    }
    Eigen::RowVector2d output = parent_center + change_vector;
    return output;
  };

  // How many cells do we have so far?
  int m = 0;
  int opt_depth = 0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> helper;
  helper = [&helper, &translate_center, &get_quad, &m, &zero_to_three, &neg_ones, 
      &P, &E, &C, &P_I, &P_EI, &P_LL, &cPE, &children, &parent, &levels, &centers, &widths,
      &max_depth, &min_pnt_num, &opt_depth](const int index, const int depth) -> void {
    if (P_I.at(index).size() > min_pnt_num && depth < max_depth) {
      //give the parent access to the children
      children.at(index) = zero_to_three.array() + m;
      //make the children's data in our arrays

      //Add the children to the lists, as default children
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
        P_EI.emplace_back(std::vector<int>());
        P_LL.emplace_back(RowVec2d_list());
        cPE.emplace_back(Mat2d_list());
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(C.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int i = 0; i < 4; i++) {
        Eigen::Matrix2d bnd;
        bnd(0,0) = centers[m+i](0)-h/2;
        bnd(1,0) = centers[m+i](0)+h/2;
        bnd(0,1) = centers[m+i](1)-h/2;
        bnd(1,1) = centers[m+i](1)+h/2;

        for (int j = 0; j < P_EI.at(index).size(); j++) {
          int curr_point_index = P_EI.at(index).at(j);
          Eigen::RowVector2d p1 = P.row(E(curr_point_index,0));
          Eigen::RowVector2d p2 = P.row(E(curr_point_index,1));

          Eigen::RowVector2d np1, np2, rns; 
          if(liang_barsky_clipper(bnd,p1,p2,np1,np2,rns)){
            P_EI.at(m+i).emplace_back(curr_point_index);
            P_LL.at(m+i).emplace_back(rns);
            Eigen::Matrix2d nps;
            nps.row(0) = np1;
            nps.row(1) = np2;
            cPE.at(m+i).emplace_back(nps);
          }
        }
      }

      //Now increase m
      m += 4;
      opt_depth = std::max(opt_depth,depth);
      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        helper(children.at(index)(i), depth + 1);
      }
    }
  };


  {
    std::vector<int> allC(C.rows());
    for (int i = 0; i < allC.size(); i++)
      allC[i] = i;
    P_I.emplace_back(allC);
    P_EI.emplace_back(allC);
    P_LL.emplace_back(RowVec2d_list());
    cPE.emplace_back(Mat2d_list());
  }

  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  helper(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }
  // LV.resize(opt_depth+1);
  // for (int i=0;i<levels.size(); i++){
  //   LV[levels[i]].emplace_back(i);
  // }

}



void infinite::quadtree(const Eigen::MatrixXd &P, 
             const Eigen::MatrixXd &Q,
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& max_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &Q_I, 
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::VectorXi &LV,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{

 std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  auto get_quad = [](const Eigen::RowVector2d &location,
    const Eigen::RowVector2d &center) {
    int index = 0;
    if (location(0) >= center(0)) {
      index = index + 1;
    }
    if (location(1) >= center(1)) {
      index = index + 2;
    }
    return index;
  };

  std::function<
      Eigen::RowVector2d(const Eigen::RowVector2d, const double,
          const int)> translate_center = [](
      const Eigen::RowVector2d &parent_center, const double h,
      const int child_index) {
    Eigen::RowVector2d change_vector;
    change_vector << -h, -h;

    //positive x chilren are 1,3
    if (child_index % 2) {
      change_vector(0) = h;
    }
    //positive y children are 2,3
    if (child_index == 2 || child_index == 3) {
      change_vector(1) = h;
    }
    Eigen::RowVector2d output = parent_center + change_vector;
    return output;
  };

  // How many cells do we have so far?
  int m = 0;
  int opt_depth=0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> helper;
  helper = [&helper, &translate_center, &get_quad, &m, &zero_to_three,
      &neg_ones, &P, &Q, &P_I, &Q_I, &children, &parent, &levels, &centers, &widths,
      &max_depth, &min_pnt_num, &opt_depth](const int index, const int depth) -> void {
    if (P_I.at(index).size() > min_pnt_num && depth < max_depth) {
      //give the parent access to the children
      children.at(index) = zero_to_three.array() + m;
      //make the children's data in our arrays

      //Add the children to the lists, as default children
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        Q_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(P.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(Q.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      //Now increase m
      m += 4;
      opt_depth = std::max(opt_depth,depth);

      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        helper(children.at(index)(i), depth + 1);
      }
    }
  };


  {
    std::vector<int> allP(P.rows());
    for (int i = 0; i < allP.size(); i++)
      allP[i] = i;
    std::vector<int> allQ(Q.rows());
    for (int i = 0; i < allQ.size(); i++)
      allQ[i] = i;

    P_I.emplace_back(allP);
    Q_I.emplace_back(allQ);
  }
  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  helper(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }

}


void infinite::quadtree(const Eigen::MatrixXd &P, 
             const Eigen::MatrixXd &Q,
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& max_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &Q_I, 
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             std::vector<std::vector<int> > &Lv,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{

 std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  auto get_quad = [](const Eigen::RowVector2d &location,
    const Eigen::RowVector2d &center) {
    int index = 0;
    if (location(0) >= center(0)) {
      index = index + 1;
    }
    if (location(1) >= center(1)) {
      index = index + 2;
    }
    return index;
  };

  std::function<
      Eigen::RowVector2d(const Eigen::RowVector2d, const double,
          const int)> translate_center = [](
      const Eigen::RowVector2d &parent_center, const double h,
      const int child_index) {
    Eigen::RowVector2d change_vector;
    change_vector << -h, -h;

    //positive x chilren are 1,3
    if (child_index % 2) {
      change_vector(0) = h;
    }
    //positive y children are 2,3
    if (child_index == 2 || child_index == 3) {
      change_vector(1) = h;
    }
    Eigen::RowVector2d output = parent_center + change_vector;
    return output;
  };

  // How many cells do we have so far?
  int m = 0;
  int opt_depth=0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> helper;
  helper = [&helper, &translate_center, &get_quad, &m, &zero_to_three,
      &neg_ones, &P, &Q, &P_I, &Q_I, &children, &parent, &levels, &centers, &widths,
      &max_depth, &min_pnt_num, &opt_depth](const int index, const int depth) -> void {
    if (P_I.at(index).size() > min_pnt_num && depth < max_depth) {
      //give the parent access to the children
      children.at(index) = zero_to_three.array() + m;
      //make the children's data in our arrays

      //Add the children to the lists, as default children
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        Q_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(P.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(Q.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      //Now increase m
      m += 4;
      opt_depth = std::max(opt_depth,depth);

      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        helper(children.at(index)(i), depth + 1);
      }
    }
  };


  {
    std::vector<int> allP(P.rows());
    for (int i = 0; i < allP.size(); i++)
      allP[i] = i;
    std::vector<int> allQ(Q.rows());
    for (int i = 0; i < allQ.size(); i++)
      allQ[i] = i;

    P_I.emplace_back(allP);
    Q_I.emplace_back(allQ);
  }
  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  helper(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }

  Lv.resize(opt_depth+1);
  for (int i=0;i<levels.size(); i++){
    Lv[levels[i]].emplace_back(i);
  }

}



void infinite::quadtree(const Eigen::MatrixXd &P, 
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& max_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &point_indices, 
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             std::vector<std::vector<int> > &Lv,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{

 std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  auto get_quad = [](const Eigen::RowVector2d &location,
    const Eigen::RowVector2d &center) {
    int index = 0;
    if (location(0) >= center(0)) {
      index = index + 1;
    }
    if (location(1) >= center(1)) {
      index = index + 2;
    }
    return index;
  };

  std::function<
      Eigen::RowVector2d(const Eigen::RowVector2d, const double,
          const int)> translate_center = [](
      const Eigen::RowVector2d &parent_center, const double h,
      const int child_index) {
    Eigen::RowVector2d change_vector;
    change_vector << -h, -h;

    //positive x chilren are 1,3
    if (child_index % 2) {
      change_vector(0) = h;
    }
    //positive y children are 2,3
    if (child_index == 2 || child_index == 3) {
      change_vector(1) = h;
    }
    Eigen::RowVector2d output = parent_center + change_vector;
    return output;
  };

  // How many cells do we have so far?
  int m = 0;
  int opt_depth = 0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> helper;
  helper = [&helper, &translate_center, &get_quad, &m, &zero_to_three,
      &neg_ones, &P, &point_indices, &children, &parent, &levels, &centers, &widths,
      &max_depth, &min_pnt_num, &opt_depth](const int index, const int depth) -> void {
    if (point_indices.at(index).size() > min_pnt_num && depth < max_depth) {
      //give the parent access to the children
      children.at(index) = zero_to_three.array() + m;
      //make the children's data in our arrays

      //Add the children to the lists, as default children
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        point_indices.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < point_indices.at(index).size(); j++) {
        int curr_point_index = point_indices.at(index).at(j);
        int cell_of_curr_point = get_quad(P.row(curr_point_index),
            curr_center) + m;
        point_indices.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }

      //Now increase m
      m += 4;
      opt_depth = std::max(opt_depth,depth);
      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        helper(children.at(index)(i), depth + 1);
      }
    }
  };


  {
    std::vector<int> all(P.rows());
    for (int i = 0; i < all.size(); i++)
      all[i] = i;
    point_indices.emplace_back(all);
  }
  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  helper(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }


  Lv.resize(opt_depth+1);
  for (int i=0;i<levels.size(); i++){
    Lv[levels[i]].emplace_back(i);
  }

}



void quadtree_uniform(const Eigen::MatrixXd &P, 
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& max_depth,
             const int& min_pnt_num,
             std::vector<std::vector<int> > &point_indices, 
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{

 std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;

  std::vector<int> parent;

  std::vector<double> widths;

  auto get_quad = [](const Eigen::RowVector2d &location,
    const Eigen::RowVector2d &center) {
    int index = 0;
    if (location(0) >= center(0)) {
      index = index + 1;
    }
    if (location(1) >= center(1)) {
      index = index + 2;
    }
    return index;
  };

  std::function<
      Eigen::RowVector2d(const Eigen::RowVector2d, const double,
          const int)> translate_center = [](
      const Eigen::RowVector2d &parent_center, const double h,
      const int child_index) {
    Eigen::RowVector2d change_vector;
    change_vector << -h, -h;

    //positive x chilren are 1,3
    if (child_index % 2) {
      change_vector(0) = h;
    }
    //positive y children are 2,3
    if (child_index == 2 || child_index == 3) {
      change_vector(1) = h;
    }
    Eigen::RowVector2d output = parent_center + change_vector;
    return output;
  };

  // How many cells do we have so far?
  int m = 0;

  int opt_depth = 0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> depth_test;
  depth_test = [&depth_test, &translate_center, &get_quad, &m, &zero_to_three,
      &neg_ones, &P, &point_indices, &children, &centers, &widths,
      &max_depth, &min_pnt_num, &opt_depth](const int index, const int depth) -> void {
    if (point_indices.at(index).size() > min_pnt_num && depth < max_depth) {
      //give the parent access to the children
      children.at(index) = zero_to_three.array() + m;
      //make the children's data in our arrays

      //Add the children to the lists, as default children
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        point_indices.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < point_indices.at(index).size(); j++) {
        int curr_point_index = point_indices.at(index).at(j);
        int cell_of_curr_point = get_quad(P.row(curr_point_index),
            curr_center) + m;
        point_indices.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }

      //Now increase m
      m += 4;
      opt_depth = depth;
      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        depth_test(children.at(index)(i), depth + 1);
      }
    }
  };


  std::function<void(const int, const int)> helper;
  helper = [&helper, &translate_center, &get_quad, &m, &zero_to_three,
      &neg_ones, &P, &point_indices, &children, &parent, &centers, &widths,
      &opt_depth](const int index, const int depth) -> void {
    if (depth <= opt_depth) {
      //give the parent access to the children
      children.at(index) = zero_to_three.array() + m;
      //make the children's data in our arrays

      //Add the children to the lists, as default children
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        point_indices.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < point_indices.at(index).size(); j++) {
        int curr_point_index = point_indices.at(index).at(j);
        int cell_of_curr_point = get_quad(P.row(curr_point_index),
            curr_center) + m;
        point_indices.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }

      //Now increase m
      m += 4;

      // Look ma, I'm calling myself.
      for (int i = 0; i < 4; i++) {
        helper(children.at(index)(i), depth + 1);
      }
    }
  };



  {
    std::vector<int> all(P.rows());
    for (int i = 0; i < all.size(); i++)
      all[i] = i;
    point_indices.emplace_back(all);
  }
  children.emplace_back(neg_ones);
  parent.emplace_back(-1);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);
  widths.emplace_back(aabb_width);
  m++;

  // then you have to actually call the function

  depth_test(0,0);

  //std::cout<<"opt_depth: "<<opt_depth<<std::endl;

  children.clear();
  parent.clear();
  centers.clear();
  widths.clear();
  point_indices.clear();

  {
    std::vector<int> all(P.rows());
    for (int i = 0; i < all.size(); i++)
      all[i] = i;
    point_indices.emplace_back(all);
  }
  children.emplace_back(neg_ones);
  parent.emplace_back(-1);
  centers.emplace_back(aabb_center);
  widths.emplace_back(aabb_width);
  m=1;

  helper(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
}




void infinite::quadtree_uniform(const Eigen::MatrixXd &P, 
             const Eigen::MatrixXi &E,
             const Eigen::MatrixXd &C, 
             const Eigen::MatrixXd &Q,
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& set_depth,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &P_EI,
             std::vector<std::vector<int> > &Q_I,
             RowVec2d_list_list &P_LL,
             Mat2d_list_list &PP,
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::VectorXi &LV,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{
  double s = igl::get_seconds();
  std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  int m = 0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> build_quadtree;
  build_quadtree = [&build_quadtree,&m,&zero_to_three,&neg_ones,&P,&E,&C,&Q, 
      &P_I,&P_EI,&Q_I,&P_LL,&PP,&children,&parent,&levels, 
      &centers,&widths,&set_depth]
      (const int index, const int depth) -> void {
    if (depth <= set_depth) {

      children.at(index) = zero_to_three.array() + m;
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        Q_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
        P_EI.emplace_back(std::vector<int>());
        P_LL.emplace_back(RowVec2d_list());
        PP.emplace_back(Mat2d_list());
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(C.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(Q.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }

      for (int i = 0; i < 4; i++) {
        Eigen::Matrix2d bnd;
        bnd(0,0) = centers[m+i](0)-h/2;
        bnd(1,0) = centers[m+i](0)+h/2;
        bnd(0,1) = centers[m+i](1)-h/2;
        bnd(1,1) = centers[m+i](1)+h/2;

        for (int j = 0; j < P_EI.at(index).size(); j++) {
          int curr_point_index = P_EI.at(index).at(j);
          Eigen::RowVector2d p1 = P.row(E(curr_point_index,0));
          Eigen::RowVector2d p2 = P.row(E(curr_point_index,1));

          Eigen::RowVector2d np1, np2, rns; 
          if(liang_barsky_clipper(bnd,p1,p2,np1,np2,rns)){
            P_EI.at(m+i).emplace_back(curr_point_index);
            P_LL.at(m+i).emplace_back(rns);
            Eigen::Matrix2d nps;
            nps.row(0) = np1;
            nps.row(1) = np2;
            PP.at(m+i).emplace_back(nps);
          }
        }
      }
      //Now increase m
      m += 4;
      for (int i = 0; i < 4; i++) {
        build_quadtree(children.at(index)(i), depth + 1);
      }
    }
  };

  {
    std::vector<int> allC(C.rows());
    for (int i = 0; i < allC.size(); i++)
      allC[i] = i;
    std::vector<int> allQ(Q.rows());
    for (int i = 0; i < allQ.size(); i++)
      allQ[i] = i;
    P_I.emplace_back(allC);
    P_EI.emplace_back(allC);    
    Q_I.emplace_back(allQ);
    P_LL.emplace_back(RowVec2d_list());
    PP.emplace_back(Mat2d_list());
  }

  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  build_quadtree(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }
  // LV.resize(opt_depth+1);
  // for (int i=0;i<levels.size(); i++){
  //   LV[levels[i]].emplace_back(i);
  // }
  double t = igl::get_seconds();
}


void infinite::quadtree_uniform(const Eigen::MatrixXd &C, 
             const Eigen::MatrixXd &Q,
             const Eigen::RowVector2d& minP, 
             const Eigen::RowVector2d& maxP,
             const int& set_depth,
             std::vector<std::vector<int> > &P_I, 
             std::vector<std::vector<int> > &Q_I,
             Eigen::MatrixXi &CH,
             Eigen::VectorXi &PA,
             Eigen::VectorXi &LV,
             Eigen::MatrixXd &CN, 
             Eigen::VectorXd &W) 
{
  double s = igl::get_seconds();
  std::vector<Eigen::Vector4i,
      Eigen::aligned_allocator<Eigen::Vector4i> > children;
  std::vector<Eigen::RowVector2d,
      Eigen::aligned_allocator<Eigen::RowVector2d> > centers;


  std::vector<int> parent;
  std::vector<int> levels;
  std::vector<double> widths;

  int m = 0;

  // Useful list of number 0,1,2,3
  const Eigen::Vector4i zero_to_three =
      (Eigen::Vector4i() << 0, 1, 2, 3).finished();
  const Eigen::Vector4i neg_ones = Eigen::Vector4i::Constant(-1);

  std::function<void(const int, const int)> build_quadtree;
  build_quadtree = [&build_quadtree,&m,&zero_to_three,&neg_ones,&C,&Q, 
      &P_I,&Q_I,&children,&parent,&levels, 
      &centers,&widths,&set_depth]
      (const int index, const int depth) -> void {
    if (depth <= set_depth) {

      children.at(index) = zero_to_three.array() + m;
      double h = widths.at(index) / 2;
      Eigen::RowVector2d curr_center = centers.at(index);

      for (int i = 0; i < 4; i++) {
        children.emplace_back(neg_ones);
        P_I.emplace_back(std::vector<int>());
        Q_I.emplace_back(std::vector<int>());
        centers.emplace_back(translate_center(curr_center, h / 2, i));
        widths.emplace_back(h);
        parent.emplace_back(index);
        levels.emplace_back(depth);
      }

      //Split up the points into the corresponding children
      for (int j = 0; j < P_I.at(index).size(); j++) {
        int curr_point_index = P_I.at(index).at(j);
        int cell_of_curr_point = get_quad(C.row(curr_point_index),
            curr_center) + m;
        P_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      for (int j = 0; j < Q_I.at(index).size(); j++) {
        int curr_point_index = Q_I.at(index).at(j);
        int cell_of_curr_point = get_quad(Q.row(curr_point_index),
            curr_center) + m;
        Q_I.at(cell_of_curr_point).emplace_back(
            curr_point_index);
      }
      //Now increase m
      m += 4;
      for (int i = 0; i < 4; i++) {
        build_quadtree(children.at(index)(i), depth + 1);
      }
    }
  };

  {
    std::vector<int> allC(C.rows());
    for (int i = 0; i < allC.size(); i++)
      allC[i] = i;
    std::vector<int> allQ(Q.rows());
    for (int i = 0; i < allQ.size(); i++)
      allQ[i] = i;
    P_I.emplace_back(allC);
    Q_I.emplace_back(allQ);
  }

  children.emplace_back(neg_ones);
  parent.emplace_back(-2);
  levels.emplace_back(0);

  //Get the minimum AABB for the points
  Eigen::RowVector2d aabb_center = (minP + maxP) / double(2.0);
  double aabb_width = (maxP - minP).maxCoeff();
  centers.emplace_back(aabb_center);

  //Widths are the side length of the cube, (not half the side length):
  widths.emplace_back(aabb_width);
  m++;
  // then you have to actually call the function
  build_quadtree(0, 0);

  //Now convert from vectors to Eigen matricies:
  CH.resize(children.size(), 4);
  CN.resize(centers.size(), 2);
  W.resize(widths.size(), 1);
  PA.resize(parent.size(),1);
  LV.resize(levels.size(),1);

  for (int i = 0; i < children.size(); i++) {
    CH.row(i) = children.at(i);
  }
  for (int i = 0; i < centers.size(); i++) {
    CN.row(i) = centers.at(i);
  }
  for (int i = 0; i < widths.size(); i++) {
    W(i) = widths.at(i);
  }
  for (int i = 0; i < parent.size(); i++) {
    PA(i) = parent.at(i);
  }
  for (int i = 0; i < levels.size(); i++) {
    LV(i) = levels.at(i);
  }
  // LV.resize(opt_depth+1);
  // for (int i=0;i<levels.size(); i++){
  //   LV[levels[i]].emplace_back(i);
  // }
  double t = igl::get_seconds();
}


