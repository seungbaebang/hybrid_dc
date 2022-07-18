#include "expansion.h"

#include <iostream>


void infinite::update_outgoing_from_source_g(const std::vector<bool> &update_pids,
                      const Eigen::VectorXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mg)
{
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int cell_id = leaf_cells[i];
    const std::vector<int> &ids = PI[cell_id];
    Mg.row(cell_id)=Eigen::RowVectorXcd::Zero(np);
    for(int j=0; j<ids.size(); j++){
      if(!update_pids[ids[j]])
        continue;
      Mg.row(cell_id)=Mg.row(cell_id)+Ik_out_list[cell_id][j].transpose()*sigma(ids[j]);
    }
  }
}

void infinite::update_outgoing_from_source_f(const std::vector<bool> &update_pids,
                      const Eigen::VectorXd &mu,
                      const Eigen::VectorXcd &Ni,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mf)
{
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int cell_id = leaf_cells[i];
    const std::vector<int> &ids = PI[cell_id];
    Mf.row(cell_id)=Eigen::RowVectorXcd::Zero(np);
    for(int j=0; j<ids.size(); j++){
      if(!update_pids[ids[j]])
        continue;
      Mf.row(cell_id)=Mf.row(cell_id)+Ni(ids[j])*Ik_out_list[cell_id][j].transpose()*mu(ids[j]);
    }
  }
}


void infinite::update_incoming_from_source_g(const std::vector<bool> &update_pids,
                        const Eigen::VectorXd &sigma,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Ug)
{
  // Ug = Eigen::MatrixXcd::Zero(Q_big_seps.size(),np);
  #pragma omp parallel for
  for(int i=0; i<Q_big_seps.size(); i++){
    for(int ii=0; ii<Q_big_seps[i].size(); ii++){
      int big_sep_cell = Q_big_seps[i][ii];
      for(int jj=0; jj<Q_PEI[big_sep_cell].size(); jj++){
        int pid = Q_PEI[big_sep_cell][jj];
        if(!update_pids[pid])
          continue;
        Eigen::RowVectorXcd Ok = Ok_inc_list_list[i][ii][jj].transpose();
        Ug.row(i) = Ug.row(i) - Ok.head(np)*sigma(pid);
      }
    }
  }
}


void infinite::update_incoming_from_sourc_f(const std::vector<bool> &update_pids,
                        const Eigen::VectorXd &mu,
                        const Eigen::VectorXcd &Ni,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Uf)
{
  // Uf = Eigen::MatrixXcd::Zero(Q_big_seps.size(),np);
  #pragma omp parallel for
  for(int i=0; i<Q_big_seps.size(); i++){
    for(int ii=0; ii<Q_big_seps[i].size(); ii++){
      int big_sep_cell = Q_big_seps[i][ii];
      for(int jj=0; jj<Q_PEI[big_sep_cell].size(); jj++){
        int pid = Q_PEI[big_sep_cell][jj];
        if(!update_pids[pid])
          continue;
        Eigen::RowVectorXcd Ok = Ok_inc_list_list[i][ii][jj].transpose();
        Uf.row(i) = Uf.row(i) + Ni(pid)*Ok.tail(np)*mu(pid);
      }
    }
  }
}



void infinite::incoming_from_outgoing(const std::vector<std::vector<int> > &PI,
                  const std::vector<std::vector<int> > &Inters,
                  const Eigen::MatrixXcd &Mg,
                  const Eigen::MatrixXcd &Mf,
                  const int &np,
                  const std::vector<VecXcd_list> &Ok_inter_list_list,
                  Eigen::MatrixXcd &Lg,
                  Eigen::MatrixXcd &Lf)
{
  #pragma omp parallel for
  for(int i=0; i<Inters.size(); i++){
    const VecXcd_list &Ok = Ok_inter_list_list[i];
    for(int j=0; j<Inters[i].size(); j++){
      int ii = Inters[i][j];
      if(PI[ii].size()==0)
        continue;
      for(int l=0; l<np; l++){
        for(int k=0; k<np; k++){
          if(l%2==0){
            Lg(i,l) = Lg(i,l)+Ok[j](l+k)*Mg(ii,k);
            Lf(i,l) = Lf(i,l)+Ok[j](l+k+1)*Mf(ii,k);
          }
          else{
            Lg(i,l) = Lg(i,l)-Ok[j](l+k)*Mg(ii,k);
            Lf(i,l) = Lf(i,l)-Ok[j](l+k+1)*Mf(ii,k);
          }
        }
      }
    }
  }
}


void infinite::incoming_from_outgoing_g(const std::vector<std::vector<int> > &PI,
                  const std::vector<std::vector<int> > &Inters,
                  const Eigen::MatrixXcd &Mg,
                  const int &np,
                  const std::vector<VecXcd_list> &Ok_inter_list_list,
                  Eigen::MatrixXcd &Lg)
{
  #pragma omp parallel for
  for(int i=0; i<Inters.size(); i++){
    const VecXcd_list &Ok = Ok_inter_list_list[i];
    for(int j=0; j<Inters[i].size(); j++){
      int ii = Inters[i][j];
      if(PI[ii].size()==0)
        continue;
      for(int l=0; l<np; l++){
        for(int k=0; k<np; k++){
          if(l%2==0){
            Lg(i,l) = Lg(i,l)+Ok[j](l+k)*Mg(ii,k);
          }
          else{
            Lg(i,l) = Lg(i,l)-Ok[j](l+k)*Mg(ii,k);
          }
        }
      }
    }
  }
}


void infinite::incoming_from_outgoing_f(const std::vector<std::vector<int> > &PI,
                  const std::vector<std::vector<int> > &Inters,
                  const Eigen::MatrixXcd &Mf,
                  const int &np,
                  const std::vector<VecXcd_list> &Ok_inter_list_list,
                  Eigen::MatrixXcd &Lf)
{
  #pragma omp parallel for
  for(int i=0; i<Inters.size(); i++){
    const VecXcd_list &Ok = Ok_inter_list_list[i];
    for(int j=0; j<Inters[i].size(); j++){
      int ii = Inters[i][j];
      if(PI[ii].size()==0)
        continue;
      for(int l=0; l<np; l++){
        for(int k=0; k<np; k++){
          if(l%2==0){
            Lf(i,l) = Lf(i,l)+Ok[j](l+k+1)*Mf(ii,k);
          }
          else{
            Lf(i,l) = Lf(i,l)-Ok[j](l+k+1)*Mf(ii,k);
          }
        }
      }
    }
  }
}


void infinite::incoming_from_incoming(const std::vector<std::vector<int> > &PI,
                  const std::vector<std::vector<int> > &LV, 
                  const Eigen::MatrixX4i &CH,
                  const int &np,
                  const MatXcd_list &Ik_child_list,
                  Eigen::MatrixXcd &Lg,
                  Eigen::MatrixXcd &Lf)
{
  for(int i=0; i<LV.size(); i++){
    #pragma omp parallel for
    for(int j=0; j<LV[i].size(); j++){
      int cell_id = LV[i][j];
      if(CH(cell_id,0)<0)
        continue;
      if(PI[cell_id].size()==0)
        continue;
      const Eigen::MatrixXcd &Ik = Ik_child_list[cell_id];
      for(int ci=0; ci<4; ci++){
        int child_index = CH(cell_id,ci);
        for(int l=0; l<np; l++){
          for(int m=l; m<np; m++){
            Lg(child_index,l)=Lg(child_index,l)+Ik(ci,m-l)*Lg(cell_id,m);
            Lf(child_index,l)=Lf(child_index,l)+Ik(ci,m-l)*Lf(cell_id,m);
          }
        }
      }
    }
  }
}


void infinite::incoming_from_incoming(const std::vector<std::vector<int> > &PI,
                  const std::vector<std::vector<int> > &LV, 
                  const Eigen::MatrixX4i &CH,
                  const int &np,
                  const MatXcd_list &Ik_child_list,
                  Eigen::MatrixXcd &Lx)
{
  for(int i=0; i<LV.size(); i++){
    #pragma omp parallel for
    for(int j=0; j<LV[i].size(); j++){
      int cell_id = LV[i][j];
      if(CH(cell_id,0)<0)
        continue;
      if(PI[cell_id].size()==0)
        continue;
      const Eigen::MatrixXcd &Ik = Ik_child_list[cell_id];
      for(int ci=0; ci<4; ci++){
        int child_index = CH(cell_id,ci);
        for(int l=0; l<np; l++){
          for(int m=l; m<np; m++){
            Lx(child_index,l)=Lx(child_index,l)+Ik(ci,m-l)*Lx(cell_id,m);
          }
        }
      }
    }
  }
}







void infinite::incoming_from_source(const Eigen::VectorXd &sigma,
                        const Eigen::VectorXd &mu,
                        const Eigen::VectorXcd &Ni,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Ug,
                        Eigen::MatrixXcd &Uf)
{
  #pragma omp parallel for
  for(int i=0; i<Q_big_seps.size(); i++){
    for(int ii=0; ii<Q_big_seps[i].size(); ii++){
      int big_sep_cell = Q_big_seps[i][ii];
      for(int jj=0; jj<Q_PEI[big_sep_cell].size(); jj++){
        int pid = Q_PEI[big_sep_cell][jj];
        Eigen::RowVectorXcd Ok = Ok_inc_list_list[i][ii][jj].transpose();
        Ug.row(i) = Ug.row(i) - Ok.head(np)*sigma(pid);
        Uf.row(i) = Uf.row(i) + Ni(pid)*Ok.tail(np)*mu(pid);
      }
    }
  }
}



void infinite::incoming_from_source_g(const Eigen::VectorXd &sigma,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Ug)
{
  Ug = Eigen::MatrixXcd::Zero(Q_big_seps.size(),np);
  #pragma omp parallel for
  for(int i=0; i<Q_big_seps.size(); i++){
    for(int ii=0; ii<Q_big_seps[i].size(); ii++){
      int big_sep_cell = Q_big_seps[i][ii];
      for(int jj=0; jj<Q_PEI[big_sep_cell].size(); jj++){
        int pid = Q_PEI[big_sep_cell][jj];
        Eigen::RowVectorXcd Ok = Ok_inc_list_list[i][ii][jj].transpose();
        Ug.row(i) = Ug.row(i) - Ok.head(np)*sigma(pid);
      }
    }
  }
}


void infinite::incoming_from_source_f(const Eigen::VectorXd &mu,
                        const Eigen::VectorXcd &Ni,
                        const std::vector<std::vector<VecXcd_list> > &Ok_inc_list_list,
                        const std::vector<std::vector<int> > &Q_big_seps,
                        const std::vector<std::vector<int> > &Q_PEI,
                        const int &np,
                        Eigen::MatrixXcd &Uf)
{
  Uf = Eigen::MatrixXcd::Zero(Q_big_seps.size(),np);
  #pragma omp parallel for
  for(int i=0; i<Q_big_seps.size(); i++){
    for(int ii=0; ii<Q_big_seps[i].size(); ii++){
      int big_sep_cell = Q_big_seps[i][ii];
      for(int jj=0; jj<Q_PEI[big_sep_cell].size(); jj++){
        int pid = Q_PEI[big_sep_cell][jj];
        Eigen::RowVectorXcd Ok = Ok_inc_list_list[i][ii][jj].transpose();
        Uf.row(i) = Uf.row(i) + Ni(pid)*Ok.tail(np)*mu(pid);
      }
    }
  }
}





void infinite::outgoing_from_source(const Eigen::VectorXd &sigma,
                      const Eigen::VectorXd &mu,
                      const Eigen::VectorXcd &Ni,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mg,
                      Eigen::MatrixXcd &Mf)
{
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int cell_id = leaf_cells[i];
    const std::vector<int> &ids = PI[cell_id];
    Mg.row(cell_id)=Eigen::RowVectorXcd::Zero(np);
    Mf.row(cell_id)=Eigen::RowVectorXcd::Zero(np);
    for(int j=0; j<ids.size(); j++){
      Mg.row(cell_id)=Mg.row(cell_id)+Ik_out_list[cell_id][j].transpose()*sigma(ids[j]);
      Mf.row(cell_id)=Mf.row(cell_id)+Ni(ids[j])*Ik_out_list[cell_id][j].transpose()*mu(ids[j]);
    }
  }
}

void infinite::outgoing_from_source_g(const Eigen::VectorXd &sigma,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mg)
{
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    // std::cout<<"num thread: "<<omp_get_num_threads()<<std::endl;
    int cell_id = leaf_cells[i];
    const std::vector<int> &ids = PI[cell_id];
    Mg.row(cell_id)=Eigen::RowVectorXcd::Zero(np);
    for(int j=0; j<ids.size(); j++){
      Mg.row(cell_id)=Mg.row(cell_id)+Ik_out_list[cell_id][j].transpose()*sigma(ids[j]);
    }
  }
}



void infinite::outgoing_from_source_f(const Eigen::VectorXd &mu,
                      const Eigen::VectorXcd &Ni,
                      const std::vector<int> &leaf_cells,
                      const std::vector<std::vector<int> > &PI,
                      const std::vector<VecXcd_list> &Ik_out_list,
                      const int &np,
                      Eigen::MatrixXcd &Mf)
{
  #pragma omp parallel for
  for(int i=0; i<leaf_cells.size(); i++){
    int cell_id = leaf_cells[i];
    const std::vector<int> &ids = PI[cell_id];
    Mf.row(cell_id)=Eigen::RowVectorXcd::Zero(np);
    for(int j=0; j<ids.size(); j++){
      Mf.row(cell_id)=Mf.row(cell_id)+Ni(ids[j])*Ik_out_list[cell_id][j].transpose()*mu(ids[j]);
    }
  }
}



void infinite::outgoing_from_outgoing(const std::vector<std::vector<int> > &LV,
                      const std::vector<std::vector<int> > &PI,
                      const MatXcd_list &Ik_child_list,
                      const Eigen::MatrixX4i &CH,
                      const int &np,
                      Eigen::MatrixXcd &Mg,
                      Eigen::MatrixXcd &Mf)
{
  for(int i=LV.size()-2; i>=0; i--){
    #pragma omp parallel for
    for(int j=0; j<LV[i].size(); j++){
      int cell_id = LV[i][j];
      if(CH(cell_id,0)<0)
        continue;
      Mf.row(cell_id) = Eigen::RowVectorXcd::Zero(np);
      Mg.row(cell_id) = Eigen::RowVectorXcd::Zero(np);
      const Eigen::MatrixXcd &Ik = Ik_child_list[cell_id];
      for(int ci=0; ci<4; ci++){
        int child_index = CH(cell_id,ci);
        if(PI[child_index].size()==0)
          continue;
        for(int k=0; k<np; k++){
          for(int l=0; l<=k; l++){
            Mf(cell_id,k) = Mf(cell_id,k) + Ik(ci,k-l)*Mf(child_index,l);
            Mg(cell_id,k) = Mg(cell_id,k) + Ik(ci,k-l)*Mg(child_index,l);
          }
        }
      }
    }
  }
}


void infinite::outgoing_from_outgoing(const std::vector<std::vector<int> > &LV,
                      const std::vector<std::vector<int> > &PI,
                      const MatXcd_list &Ik_child_list,
                      const Eigen::MatrixX4i &CH,
                      const int &np,
                      Eigen::MatrixXcd &Mx)
{
  for(int i=LV.size()-2; i>=0; i--){
    #pragma omp parallel for
    for(int j=0; j<LV[i].size(); j++){
      int cell_id = LV[i][j];
      if(CH(cell_id,0)<0)
        continue;
      Mx.row(cell_id) = Eigen::RowVectorXcd::Zero(np);
      const Eigen::MatrixXcd &Ik = Ik_child_list[cell_id];
      for(int ci=0; ci<4; ci++){
        int child_index = CH(cell_id,ci);
        if(PI[child_index].size()==0)
          continue;
        for(int k=0; k<np; k++){
          for(int l=0; l<=k; l++){
            Mx(cell_id,k) = Mx(cell_id,k) + Ik(ci,k-l)*Mx(child_index,l);
          }
        }
      }
    }
  }
}





