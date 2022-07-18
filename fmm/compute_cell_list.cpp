#include "compute_cell_list.h"
// #include <Eigen/Core>
#include <iostream>

// Eigen::RowVector2i get_node_descent(const int &index)
// {
//   Eigen::RowVector2i node_descent;
//   if((index-1)%4==0)
//     node_descent<<-1,-1;
//   else if((index-1)%4==1)
//     node_descent<<1,-1;
//   else if((index-1)%4==2)
//     node_descent<<-1,1;
//   else if((index-1)%4==3)
//     node_descent<<1,1;

//   return node_descent;
// }

int infinite::tree_neighbour(int index,
                  Eigen::RowVector2i direction,
                  const Eigen::VectorXi &PA,
                  const Eigen::MatrixXi &CH)
{
  std::vector<Eigen::RowVector2i,
      Eigen::aligned_allocator<Eigen::RowVector2i> > neighbour_descents;

  while(true){
    if(direction.isZero() || index <0){
      break;
    }
    Eigen::RowVector2i node_descent = get_node_descent(index);
    
    Eigen::RowVector2i neighbour_descent = 
          node_descent.array()*(1-2*direction.array().abs());

    neighbour_descents.emplace_back(neighbour_descent);
    direction = (node_descent + direction)/2;
    index = PA(index);
  }
    for(int i=0; i<neighbour_descents.size(); i++)
    {
      if(index<0)
        break;
      if(CH(index,0)==-1)
        break;
      Eigen::RowVector2i ni = 
            (neighbour_descents[neighbour_descents.size()-i-1].array()+1)/2;
      int child_id = 2*ni(1)+ni(0);

      index = CH(index,child_id);
    }
    int neighbour_index = index;
    return neighbour_index;
}


void infinite::near_given_level(int neighbour_index, 
             Eigen::RowVector2i direction,
             const int &level,
             const Eigen::VectorXi &LV,
             const Eigen::MatrixXi &CH,
             const std::set<int> &all_leaf_ids,
             std::vector<int> &adjs)
{

  if(all_leaf_ids.find(neighbour_index)==all_leaf_ids.end()){
    std::vector<int> ids(1);
    ids[0] = neighbour_index;
    while (ids.size()>0){
      auto it = ids.begin();
      while(it != ids.end()){
        if(LV(*it)==level || CH(*it,0)==-1){
          adjs.emplace_back(*it);
          it = ids.erase(it);
        }
        else{
          ++it;
        }
      }
      std::vector<int> n_ids(ids.size()*4);
      for(int i=0; i<ids.size(); i++){
        for(int ci=0; ci<4; ci++){
          n_ids[4*i+ci]=CH(ids[i],ci);
        }
      }
      ids = n_ids;
      n_ids.clear();
      for(int i=0; i<ids.size(); i++){
        Eigen::RowVector2i node_descent = get_node_descent(ids[i]);
        if((node_descent.array()*direction.array() <=0).all()){
          n_ids.emplace_back(ids[i]);
        }
      }
      ids = n_ids;
    }
  }
  else{
    adjs.emplace_back(neighbour_index);
  }
}



void infinite::get_adj_seperate(int neighbour_index, 
             Eigen::RowVector2i direction,
             const Eigen::MatrixXi &CH,
             const std::set<int> &all_leaf_ids,
             std::vector<int> &adjs,
             std::vector<int> &small_seps)
{

  if(all_leaf_ids.find(neighbour_index)==all_leaf_ids.end()){
    std::vector<int> ids(1);
    ids[0] = neighbour_index;
    while (ids.size()>0){
      auto it = ids.begin();
      while(it != ids.end()){
        if(CH(*it,0)==-1){
          adjs.emplace_back(*it);
          it = ids.erase(it);
        }
        else{
          ++it;
        }
      }
      std::vector<int> n_ids(ids.size()*4);
      for(int i=0; i<ids.size(); i++){
        for(int ci=0; ci<4; ci++){
          n_ids[4*i+ci]=CH(ids[i],ci);
        }
      }
      ids = n_ids;
      n_ids.clear();
      for(int i=0; i<ids.size(); i++){
        Eigen::RowVector2i node_descent = get_node_descent(ids[i]);
        if((node_descent.array()*direction.array() <=0).all()){
          n_ids.emplace_back(ids[i]);
        }
        else{
          small_seps.emplace_back(ids[i]);
        }
      }
      ids = n_ids;
    }
  }
  else{
    adjs.emplace_back(neighbour_index);
  }
}

void infinite::compute_cell_list(const Eigen::VectorXi &PA,
                       const Eigen::MatrixXi &CH,
                       std::vector<std::vector<int> > &adjs,
                       std::vector<std::vector<int> > &inters)
{
  Eigen::MatrixX2i direction(8,2);
  int k=0;
  for(int i=0; i<3; i++){
  for(int j=0; j<3; j++){
    if(i==1 && j==1)
      continue;
    direction.row(k++)<<i-1,j-1;
  }}

  adjs.resize(PA.size());
  inters.resize(PA.size());

  for(int i=0; i<PA.size(); i++){
    std::vector<int> adj;
    for(int j=0; j<8; j++){
      int ni = tree_neighbour(i,direction.row(j),PA,CH);
      if(ni>-1){
        adj.emplace_back(ni);
      }
    }
    std::vector<int> uAdj;
    igl::unique(adj,uAdj);
    adjs[i]=uAdj;
  }
  for(int i=5; i<PA.size(); i++){
    int pi = PA(i);
    std::vector<int> inter;
    for(int j=0; j<adjs[pi].size(); j++){
      if(CH(adjs[pi][j],0)==-1){
        int ni = adjs[pi][j];
        if(std::find(adjs[i].begin(),adjs[i].end(),ni)==adjs[i].end())
          inter.emplace_back(ni);
      }
      else{
        for(int cj=0; cj<4; cj++){
          int ni = CH(adjs[pi][j],cj);
          if(std::find(adjs[i].begin(),adjs[i].end(),ni)==adjs[i].end())
            inter.emplace_back(ni);
        }
        
      }
    }
    inters[i]=inter;
  }
}



void infinite::compute_cell_list(const Eigen::VectorXi &PA,
                       const Eigen::MatrixXi &CH,
                       std::vector<std::vector<int> > &adjs,
                       std::vector<std::vector<int> > &small_seps,
                       std::vector<std::vector<int> > &big_seps,
                       std::vector<std::vector<int> > &inters)
{
  std::vector<std::vector<int> > uni_adjs;
  compute_cell_list(PA,CH,adjs,small_seps,big_seps,inters,uni_adjs);
}

void infinite::compute_cell_list(const Eigen::VectorXi &PA,
                       const Eigen::MatrixXi &CH,
                       std::vector<std::vector<int> > &adjs,
                       std::vector<std::vector<int> > &small_seps,
                       std::vector<std::vector<int> > &big_seps,
                       std::vector<std::vector<int> > &inters,
                       std::vector<std::vector<int> > &uni_adjs)
{
  Eigen::MatrixX2i direction(8,2);
  int k=0;
  for(int i=0; i<3; i++){
  for(int j=0; j<3; j++){
    if(i==1 && j==1)
      continue;
    direction.row(k++)<<i-1,j-1;
  }}

  std::vector<int> all_leaf_ids;
  for(int i=0; i<CH.rows(); i++){
    if(CH(i,0)==-1){
      all_leaf_ids.emplace_back(i);
    }
  }
  std::set<int> all_leaf_set(all_leaf_ids.begin(), all_leaf_ids.end());

  uni_adjs.resize(PA.size());
  adjs.resize(PA.size());
  small_seps.resize(PA.size());
  big_seps.resize(PA.size());
  inters.resize(PA.size());

  // #pragma omp parallel for
  for(int i=1; i<PA.size(); i++){
    std::vector<int> uni_adj, adj, small_sep;
    for(int j=0; j<8; j++){
      int ni = tree_neighbour(i,direction.row(j),PA,CH);
      if(ni<=0)
        continue;
      uni_adj.emplace_back(ni);
      if(all_leaf_set.find(i)!=all_leaf_set.end()){
        std::vector<int> adj_ids, small_sep_ids;
        get_adj_seperate(ni,direction.row(j),CH,all_leaf_set,adj_ids,small_sep_ids);
        if(adj_ids.size()>0)
          adj.insert(adj.end(),adj_ids.begin(),adj_ids.end());
        if(small_sep_ids.size()>0)
          small_sep.insert(small_sep.end(),small_sep_ids.begin(),small_sep_ids.end());
      }
    }
    std::vector<int> uAdj;
    igl::unique(uni_adj,uAdj);
    uni_adjs[i]=uAdj;
    if(all_leaf_set.find(i)!=all_leaf_set.end()){
      std::vector<int> uAdj, u_small_sep;
      igl::unique(adj,uAdj);
      igl::unique(small_sep,u_small_sep);
      adjs[i]=uAdj;
      small_seps[i]=u_small_sep;
      for(int kk=0; kk<u_small_sep.size(); kk++){
        big_seps[u_small_sep[kk]].emplace_back(i);
      }
    }
  }

  // #pragma omp parallel for
  for(int i=5; i<PA.size(); i++){
    int pi = PA(i);
    std::vector<int> inter;
    for(int j=0; j<uni_adjs[pi].size(); j++){
      if(CH(uni_adjs[pi][j],0)==-1)
        continue;
      for(int cj=0; cj<4; cj++){
        int ni = CH(uni_adjs[pi][j],cj);
        if(std::find(uni_adjs[i].begin(),uni_adjs[i].end(),ni)==uni_adjs[i].end())
          inter.emplace_back(ni);
      }
    }
    inters[i]=inter;
  }
}



void infinite::update_cell_list(const Eigen::VectorXi &sub_cells,
                        const Eigen::VectorXi &new_cells,
                        const Eigen::MatrixXi &CH,
                        const Eigen::VectorXi &PA,
                        const Eigen::VectorXi &LV,
                        const std::set<int> &all_leaf_set,
                        std::vector<std::vector<int> > &adjs,
                        std::vector<std::vector<int> > &small_seps,
                        std::vector<std::vector<int> > &big_seps,
                        std::vector<std::vector<int> > &inters,
                        std::vector<std::vector<int> > &uni_adjs)
{
  Eigen::MatrixX2i direction(8,2);
  int k=0;
  for(int i=0; i<3; i++){
  for(int j=0; j<3; j++){
    if(i==1 && j==1)
      continue;
    direction.row(k++)<<i-1,j-1;
  }}

  for(int i=0; i<new_cells.size(); i++){
    adjs.emplace_back(std::vector<int>());
    small_seps.emplace_back(std::vector<int>());
    big_seps.emplace_back(std::vector<int>());
    inters.emplace_back(std::vector<int>());
    uni_adjs.emplace_back(std::vector<int>());
  }


  //update new cells
  for(int i=0; i<new_cells.size(); i++){
    int new_cell = new_cells[i];

    std::vector<int> uni_adj, adj, small_sep;
    for(int j=0; j<8; j++){
      int ni = tree_neighbour(new_cell,direction.row(j),PA,CH);
      if(ni<=0)
        continue;
      uni_adj.emplace_back(ni);
      if(all_leaf_set.find(new_cell)!=all_leaf_set.end()){
        std::vector<int> adj_ids, small_sep_ids;
        get_adj_seperate(ni,direction.row(j),CH,all_leaf_set,adj_ids,small_sep_ids);
        if(adj_ids.size()>0)
          adj.insert(adj.end(),adj_ids.begin(),adj_ids.end());
        if(small_sep_ids.size()>0)
          small_sep.insert(small_sep.end(),small_sep_ids.begin(),small_sep_ids.end());
      }
    }
    std::vector<int> uAdj;
    igl::unique(uni_adj,uAdj);
    uni_adjs[new_cell]=uAdj;


    // uni_adjs.emplace_back(uAdj);
    if(all_leaf_set.find(new_cell)!=all_leaf_set.end()){
      std::vector<int> uAdj, u_small_sep;
      igl::unique(adj,uAdj);
      igl::unique(small_sep,u_small_sep);
      adjs[new_cell]=uAdj;
      small_seps[new_cell]=u_small_sep;
      for(int kk=0; kk<u_small_sep.size(); kk++){
        if(std::find(big_seps[u_small_sep[kk]].begin(),big_seps[u_small_sep[kk]].end(),new_cell)==big_seps[u_small_sep[kk]].end())
          big_seps[u_small_sep[kk]].emplace_back(new_cell);
      }
    }
  }


  std::function<void(const int, std::vector<int>&)> collect_child_cells;
  collect_child_cells = [&collect_child_cells,&CH]
  (const int index, std::vector<int>& list)->void{
    if(CH(index,0)==-1){
      return;
    }
    for(int ci=0;ci<4;ci++){
      list.emplace_back(CH(index,ci));
      collect_child_cells(CH(index,ci),list);
    }
  };



  //update cells that include sub cells(parent of new cells from original quadtree)
  for(int i=0; i<sub_cells.size(); i++){
    int sub_cell = sub_cells[i];
    //update adj list
    for(int j=0; j<adjs[sub_cell].size(); j++){
      int adj_cell = adjs[sub_cell][j];
      while(LV(adj_cell)>LV(sub_cell)){
        Eigen::RowVector2i direction = query_direction(adj_cell,sub_cell,PA,LV);
        std::vector<int> search_ids;
        near_given_level(sub_cell,direction,LV(adj_cell),LV,CH,all_leaf_set,search_ids);
        for(int jj=0; jj<search_ids.size(); jj++){
          Eigen::RowVector2i direction = query_direction(adj_cell,search_ids[jj],PA,LV);
          int ni = tree_neighbour(adj_cell,direction,PA,CH);
          if(ni<=0)
            continue;
          if(std::find(uni_adjs[adj_cell].begin(),uni_adjs[adj_cell].end(),ni)==uni_adjs[adj_cell].end())
            uni_adjs[adj_cell].emplace_back(ni);
          if(all_leaf_set.find(sub_cell)==all_leaf_set.end())
            uni_adjs[adj_cell].erase(std::remove(uni_adjs[adj_cell].begin(),uni_adjs[adj_cell].end(),sub_cell),uni_adjs[adj_cell].end());
        }
        adj_cell=PA(adj_cell);
      }
    }
    //update near list
    for(int j=0; j<adjs[sub_cell].size(); j++){
      int adj_cell = adjs[sub_cell][j];
      if(all_leaf_set.find(adj_cell)==all_leaf_set.end())
        continue;
      Eigen::RowVector2i direction = query_direction(adj_cell,sub_cell,PA,LV);
      std::vector<int> search_ids;
      near_given_level(sub_cell,direction,LV(adj_cell),LV,CH,all_leaf_set,search_ids);
      for(int jj=0; jj<search_ids.size(); jj++){
        Eigen::RowVector2i direction = query_direction(adj_cell,search_ids[jj],PA,LV);
        int ni = tree_neighbour(adj_cell,direction,PA,CH);
        if(ni<=0)
          continue;
        std::vector<int> adj_ids, small_sep_ids;
        get_adj_seperate(ni,direction,CH,all_leaf_set,adj_ids,small_sep_ids);
        adjs[adj_cell].erase(std::remove(adjs[adj_cell].begin(),adjs[adj_cell].end(),sub_cell),adjs[adj_cell].end());

        for(int kk=0; kk<adj_ids.size(); kk++){
          if(std::find(adjs[adj_cell].begin(),adjs[adj_cell].end(),adj_ids[kk])==adjs[adj_cell].end())
            adjs[adj_cell].emplace_back(adj_ids[kk]);
        }
        for(int kk=0; kk<small_sep_ids.size(); kk++){
          if(std::find(small_seps[adj_cell].begin(),small_seps[adj_cell].end(),small_sep_ids[kk])==small_seps[adj_cell].end())
            small_seps[adj_cell].emplace_back(small_sep_ids[kk]);
          if(std::find(big_seps[small_sep_ids[kk]].begin(),big_seps[small_sep_ids[kk]].end(),adj_cell)==big_seps[small_sep_ids[kk]].end())
            big_seps[small_sep_ids[kk]].emplace_back(adj_cell);
          
        }
      }
    }

    for(int j=0; j<small_seps[sub_cell].size(); j++){
      if(all_leaf_set.find(sub_cell)==all_leaf_set.end()){
        int cell = small_seps[sub_cell][j];
        big_seps[cell].erase(std::remove(big_seps[cell].begin(),big_seps[cell].end(),sub_cell),big_seps[cell].end());
      }
    }
  }
  for(int i=0; i<sub_cells.size(); i++){
    if(all_leaf_set.find(sub_cells[i])==all_leaf_set.end()){
      adjs[sub_cells[i]].clear();
      small_seps[sub_cells[i]].clear();
    }
  }

  //update interaction list
  for(int i=0; i<new_cells.size(); i++){
    int new_cell = new_cells[i];
    int pi = PA(new_cell);
    std::vector<int> inter;
    for(int j=0; j<uni_adjs[pi].size(); j++){
      if(CH(uni_adjs[pi][j],0)==-1)
        continue;
      for(int cj=0; cj<4; cj++){
        int ni = CH(uni_adjs[pi][j],cj);
        if(std::find(uni_adjs[new_cell].begin(),uni_adjs[new_cell].end(),ni)==uni_adjs[new_cell].end()){
          if(LV(new_cell)==LV(ni)){
            inter.emplace_back(ni);
            if(std::find(inters[ni].begin(),inters[ni].end(),new_cell)==inters[ni].end())
              inters[ni].emplace_back(new_cell);
          }
        }
      }
    }
    inters[new_cell]=inter;
  }
}


void infinite::update_cell_list(const Eigen::VectorXi &sub_cells,
                        const Eigen::VectorXi &new_cells,
                        const Eigen::MatrixXi &CH,
                        const Eigen::VectorXi &PA,
                        const Eigen::VectorXi &LV,
                        const std::set<int> &all_leaf_set,
                        std::vector<std::vector<int> > &adjs,
                        std::vector<std::vector<int> > &small_seps,
                        std::vector<std::vector<int> > &big_seps,
                        std::vector<std::vector<int> > &inters,
                        std::vector<std::vector<int> > &uni_adjs,
                        std::vector<std::vector<int> > &Inters_a,
                        std::vector<int> &big_seps_update)
{
  std::cout<<"update cell_list2 "<<std::endl;
  Eigen::MatrixX2i direction(8,2);
  int k=0;
  for(int i=0; i<3; i++){
  for(int j=0; j<3; j++){
    if(i==1 && j==1)
      continue;
    direction.row(k++)<<i-1,j-1;
  }}

  //add new cells to origial data
  for(int i=0; i<new_cells.size(); i++){
    adjs.emplace_back(std::vector<int>());
    small_seps.emplace_back(std::vector<int>());
    big_seps.emplace_back(std::vector<int>());
    inters.emplace_back(std::vector<int>());
    uni_adjs.emplace_back(std::vector<int>());
  }

  Inters_a.resize(PA.size());



  //update new cells
  for(int i=0; i<new_cells.size(); i++){
    int new_cell = new_cells[i];

    std::vector<int> uni_adj, adj, small_sep;
    for(int j=0; j<8; j++){
      int ni = tree_neighbour(new_cell,direction.row(j),PA,CH);
      if(ni<=0)
        continue;
      uni_adj.emplace_back(ni);
      if(all_leaf_set.find(new_cell)!=all_leaf_set.end()){
        std::vector<int> adj_ids, small_sep_ids;
        get_adj_seperate(ni,direction.row(j),CH,all_leaf_set,adj_ids,small_sep_ids);
        if(adj_ids.size()>0)
          adj.insert(adj.end(),adj_ids.begin(),adj_ids.end());
        if(small_sep_ids.size()>0)
          small_sep.insert(small_sep.end(),small_sep_ids.begin(),small_sep_ids.end());
      }
    }
    std::vector<int> uAdj;
    igl::unique(uni_adj,uAdj);
    uni_adjs[new_cell]=uAdj;


    // uni_adjs.emplace_back(uAdj);
    if(all_leaf_set.find(new_cell)!=all_leaf_set.end()){
      std::vector<int> uAdj, u_small_sep;
      igl::unique(adj,uAdj);
      igl::unique(small_sep,u_small_sep);
      adjs[new_cell]=uAdj;
      small_seps[new_cell]=u_small_sep;
      // Nears_a[new_cell]=uAdj;
      // L3s_a[new_cell]=u_small_sep;
      for(int kk=0; kk<u_small_sep.size(); kk++){
        if(std::find(big_seps[u_small_sep[kk]].begin(),big_seps[u_small_sep[kk]].end(),new_cell)==big_seps[u_small_sep[kk]].end()){
          big_seps[u_small_sep[kk]].emplace_back(new_cell);
          if(std::find(big_seps_update.begin(),big_seps_update.end(),u_small_sep[kk])==big_seps_update.end())
            big_seps_update.emplace_back(u_small_sep[kk]);
        }
      }
    }
  }


  std::function<void(const int, std::vector<int>&)> collect_child_cells;
  collect_child_cells = [&collect_child_cells,&CH]
  (const int index, std::vector<int>& list)->void{
    if(CH(index,0)==-1){
      return;
    }
    for(int ci=0;ci<4;ci++){
      list.emplace_back(CH(index,ci));
      collect_child_cells(CH(index,ci),list);
    }
  };



  //update cells that include sub cells(parent of new cells from original quadtree)
  for(int i=0; i<sub_cells.size(); i++){
    int sub_cell = sub_cells[i];
    //update adj list
    for(int j=0; j<adjs[sub_cell].size(); j++){
      int adj_cell = adjs[sub_cell][j];
      while(LV(adj_cell)>LV(sub_cell)){
        Eigen::RowVector2i direction = query_direction(adj_cell,sub_cell,PA,LV);
        std::vector<int> search_ids;
        near_given_level(sub_cell,direction,LV(adj_cell),LV,CH,all_leaf_set,search_ids);
        for(int jj=0; jj<search_ids.size(); jj++){
          Eigen::RowVector2i direction = query_direction(adj_cell,search_ids[jj],PA,LV);
          int ni = tree_neighbour(adj_cell,direction,PA,CH);
          if(ni<=0)
            continue;
          if(std::find(uni_adjs[adj_cell].begin(),uni_adjs[adj_cell].end(),ni)==uni_adjs[adj_cell].end()){
            uni_adjs[adj_cell].emplace_back(ni);
          }
          if(all_leaf_set.find(sub_cell)==all_leaf_set.end()){
            uni_adjs[adj_cell].erase(std::remove(uni_adjs[adj_cell].begin(),uni_adjs[adj_cell].end(),sub_cell),uni_adjs[adj_cell].end());
          }
        }
        adj_cell=PA(adj_cell);
      }
    }
    //update near list
    for(int j=0; j<adjs[sub_cell].size(); j++){
      int adj_cell = adjs[sub_cell][j];
      if(all_leaf_set.find(adj_cell)==all_leaf_set.end())
        continue;
      Eigen::RowVector2i direction = query_direction(adj_cell,sub_cell,PA,LV);
      std::vector<int> search_ids;
      near_given_level(sub_cell,direction,LV(adj_cell),LV,CH,all_leaf_set,search_ids);
      for(int jj=0; jj<search_ids.size(); jj++){
        Eigen::RowVector2i direction = query_direction(adj_cell,search_ids[jj],PA,LV);
        int ni = tree_neighbour(adj_cell,direction,PA,CH);
        if(ni<=0)
          continue;
        std::vector<int> adj_ids, small_sep_ids;
        get_adj_seperate(ni,direction,CH,all_leaf_set,adj_ids,small_sep_ids);

        if(all_leaf_set.find(sub_cell)==all_leaf_set.end()){
          adjs[adj_cell].erase(std::remove(adjs[adj_cell].begin(),adjs[adj_cell].end(),sub_cell),adjs[adj_cell].end());
          // Nears_r[adj_cell].emplace_back(sub_cell);
        }

        for(int kk=0; kk<adj_ids.size(); kk++){
          if(std::find(adjs[adj_cell].begin(),adjs[adj_cell].end(),adj_ids[kk])==adjs[adj_cell].end()){
            adjs[adj_cell].emplace_back(adj_ids[kk]);
            // Nears_a[adj_cell].emplace_back(adj_ids[kk]);
          }
        }
        for(int kk=0; kk<small_sep_ids.size(); kk++){
          if(std::find(small_seps[adj_cell].begin(),small_seps[adj_cell].end(),small_sep_ids[kk])==small_seps[adj_cell].end()){
            small_seps[adj_cell].emplace_back(small_sep_ids[kk]);
            // L3s_a[adj_cell].emplace_back(small_sep_ids[kk]);
          }
          if(std::find(big_seps[small_sep_ids[kk]].begin(),big_seps[small_sep_ids[kk]].end(),adj_cell)==big_seps[small_sep_ids[kk]].end()){
            big_seps[small_sep_ids[kk]].emplace_back(adj_cell);
            //L4s_a[small_sep_ids[kk]].emplace_back(adj_cell);
            if(std::find(big_seps_update.begin(),big_seps_update.end(),small_sep_ids[kk])==big_seps_update.end())
              big_seps_update.emplace_back(small_sep_ids[kk]);
          }
        }
      }
    }

    for(int j=0; j<small_seps[sub_cell].size(); j++){
      if(all_leaf_set.find(sub_cell)==all_leaf_set.end()){
        int cell = small_seps[sub_cell][j];
        if(std::find(big_seps[cell].begin(),big_seps[cell].end(),sub_cell)!=big_seps[cell].end()){
          big_seps[cell].erase(std::remove(big_seps[cell].begin(),big_seps[cell].end(),sub_cell),big_seps[cell].end());
          if(std::find(big_seps_update.begin(),big_seps_update.end(),cell)==big_seps_update.end())
            big_seps_update.emplace_back(cell);
        }
      }
    }
  }
  for(int i=0; i<sub_cells.size(); i++){
    if(all_leaf_set.find(sub_cells[i])==all_leaf_set.end()){
      adjs[sub_cells[i]].clear();
      small_seps[sub_cells[i]].clear();

      // Nears_r[sub_cells[i]]=adjs[sub_cells[i]];
      // L3s_r[sub_cells[i]]=small_seps[sub_cells[i]];
    }
  }

  //update interaction list
  for(int i=0; i<new_cells.size(); i++){
    int new_cell = new_cells[i];
    int pi = PA(new_cell);
    std::vector<int> inter;
    for(int j=0; j<uni_adjs[pi].size(); j++){
      if(CH(uni_adjs[pi][j],0)==-1)
        continue;
      for(int cj=0; cj<4; cj++){
        int ni = CH(uni_adjs[pi][j],cj);
        if(std::find(uni_adjs[new_cell].begin(),uni_adjs[new_cell].end(),ni)==uni_adjs[new_cell].end()){
          if(LV(new_cell)==LV(ni)){
            inter.emplace_back(ni);
            if(std::find(inters[ni].begin(),inters[ni].end(),new_cell)==inters[ni].end()){
              inters[ni].emplace_back(new_cell);
              Inters_a[ni].emplace_back(new_cell);
            }
          }
        }
      }
    }
    inters[new_cell]=inter;
    Inters_a[new_cell]=inter;
  }

}



