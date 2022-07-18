#ifndef READ_DIFFCURV_DATA_H
#define READ_DIFFCURV_DATA_H

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include "../util/types.h"


namespace infinite{

    void save_diffcurv_data(const std::string &file_name,
                        const double &image_width,
                        const double &image_height,
                        const Eigen::MatrixX2d &CP,
                        const Eigen::MatrixX4i &CE,
                        const Eigen::VectorXi &CI,
                        const MatX3d_list &CLCl,
                        const VecXd_list &CLTl,
                        const MatX3d_list &CRCl,
                        const VecXd_list &CRTl);


    void read_diffcurv_data(const std::string &file_name,
                            double &image_width,
                            double &image_height,
                            Eigen::MatrixX2d &CP,
                            Eigen::MatrixX4i &CE,
                            Eigen::VectorXi &CI,
                            MatX3d_list &CLCl,
                            VecXd_list &CLTl,
                            MatX3d_list &CRCl,
                            VecXd_list &CRTl);

    void save_diffcurv_data(const std::string &file_name,
                            double &image_width,
                            double &image_height,
                            Mat42_list &CPs,
                            MatX3d_list &CLCs,
                            VecXd_list &CLTs,
                            MatX3d_list &CRCs,
                            VecXd_list &CRTs);
                            
    void read_diffcurv_data(const std::string &file_name,
                            double &image_width,
                            double &image_height,
                            Mat42_list &CPs,
                            MatX3d_list &CLCs,
                            VecXd_list &CLTs,
                            MatX3d_list &CRCs,
                            VecXd_list &CRTs);
}

#endif