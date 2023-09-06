#ifndef BEZIER_CURVES_H
#define BEZIER_CURVES_H

#include <Eigen/Core>
#include <vector>

#include "bezier_length.h"
#include "bezier_arclen_param.h"
#include "../util/types.h"
#include "../util/edge_indices.h"

namespace infinite{

    enum curve_param {bezier=0, arclen=1};

    Eigen::RowVector2d point_bezier(const Mat42 &CP, const double& t);

    Eigen::RowVector2d normal_bezier(const Mat42 &CP, const double& t);

    Eigen::RowVector2d derivative_bezier(const Mat42 &CP, const double& t);

    Eigen::MatrixX2d points_bezier(const Mat42 &CP, const Eigen::VectorXd& T);

    Eigen::MatrixX2d normal_bezier(const Mat42 &CP, const Eigen::VectorXd& T);

    Eigen::MatrixX2d derivative_bezier(const Mat42 &CP, const Eigen::VectorXd& T);

    Eigen::MatrixX2d second_derivative_bezier(const Mat42 &CP, const Eigen::VectorXd& T);

    void line_segments_bezier(const Mat42_list &CPs,
                        const Eigen::VectorXi &nel,
                        const VecXd_list &xel,
                        const VecXd_list &xecl,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L);

    void line_segments_arclen(const Mat42_list &CPs,
                        const Eigen::VectorXi &nel,
                        const VecXd_list &xel,
                        const VecXd_list &xecl,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L);

    void line_segments_arclen(const Mat42_list &CPs,
                        const Eigen::VectorXi &nel,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L);

    void line_segments_arclen(const Mat42_list &CPs,
                        const int &ne,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L);

    void line_segments_arclen(const Mat42_list &CPs,
                        const Eigen::VectorXd &xe,
                        const Eigen::VectorXd &xec,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &C,
                        Eigen::MatrixX2d &N,
                        Eigen::MatrixX2i &E,
                        Eigen::VectorXd &L);

    void bezier_curves(const Mat42_list &CPs, 
                        const Eigen::VectorXd &T,
                        Eigen::MatrixX2d &P,
                        Eigen::MatrixX2d &N,
                        Eigen::VectorXd &CR);


    void bezier_curves_points(const Mat42_list &CPs, 
                            const Eigen::VectorXd &T,
                            const curve_param param,
                            Eigen::MatrixX2d &P);

    void bezier_curves_normals(const Mat42_list &CPs, 
                                const Eigen::VectorXd &T,
                                const curve_param param,
                                Eigen::MatrixX2d &N);


    void bezier_curves_points(const Mat42_list &CPs, 
                    const Eigen::VectorXd &T,
                    Eigen::MatrixX2d &P);

    void bezier_curves_normals(const Mat42_list &CPs, 
                                const Eigen::VectorXd &T,
                                Eigen::MatrixX2d &N);

    //////////////////////////////////////////////////////
    void bezier_curves_points(const Mat42_list &CPs, 
                            const Eigen::VectorXd &T,
                            const curve_param param,
                            MatX2d &P);

    void bezier_curves_normals(const Mat42_list &CPs, 
                                const Eigen::VectorXd &T,
                                const curve_param param,
                                MatX2d &N);

    void bezier_curves_points(const Mat42_list &CPs, 
                            const Eigen::VectorXd &T,
                            MatX2d &P);

    void bezier_curves_normals(const Mat42_list &CPs, 
                            const Eigen::VectorXd &T,
                            MatX2d &N);        

}

#endif