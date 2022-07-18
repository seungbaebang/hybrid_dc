#include "read_json.h"

bool read_json(const std::string &json_path, 
                std::string &df_curve_path,
                unsigned int &resol_height,
                unsigned int &resol_width,
                Eigen::Matrix2d &zoom_bnd,
                int &num_frame)
{
    using json = nlohmann::json;
	std::ifstream infile(json_path);
	if (!infile)
		return false;
	json j;
	infile >> j;

    if(!j.count("df_curve"))
    {
        std::cerr<<"df_curve not specified"<<std::endl;
        return false;
    }
    df_curve_path = j["df_curve"];

    resol_height = j["resol_height"];
    resol_width = j["resol_width"];
    double zoom_min_x = j["zoom_min_x"];
    double zoom_min_y = j["zoom_min_y"];
    double zoom_max_x = j["zoom_max_x"];
    double zoom_max_y = j["zoom_max_y"];
    zoom_bnd(0,0)=zoom_min_x; zoom_bnd(0,1)=zoom_min_y;
    zoom_bnd(1,0)=zoom_max_x; zoom_bnd(1,1)=zoom_max_y;
    num_frame = j["num_frame"];
}