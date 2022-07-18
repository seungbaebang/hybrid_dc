#ifndef READ_JSON_H
#define READ_JSON_H

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

#include <Eigen/Core>

bool read_json(const std::string &json_path, 
                std::string &df_curve_path,
                unsigned int &resol_height,
                unsigned int &resol_width,
                Eigen::Matrix2d &zoom_bnd,
                int &num_frame);

#endif