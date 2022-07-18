#ifndef DIFFCURV_H
#define DIFFCURV_H

#include <Eigen/Core>
#include <vector>
#include "../util/types.h"
#include "bezier_curves.h"
#include "diffcurv_color.h"
#include "subdivide_curves.h"


namespace infinite{

    void segment_diffcurv_color(const Eigen::VectorXi &dcn,
                                const MatX3d_list &CLCl,
                                const VecXd_list &CLTl,
                                const MatX3d_list &CRCl,
                                const VecXd_list &CRTl,
                                MatX3d_list &CLCs,
                                VecXd_list &CLTs,
                                MatX3d_list &CRCs,
                                VecXd_list &CRTs);

    void segment_diffcurv(const Eigen::MatrixX2d &CP,
                                const Eigen::MatrixX4i &CE,
                                const Eigen::VectorXi &CI,
                                const MatX3d_list &CLCl,
                                const VecXd_list &CLTl,
                                const MatX3d_list &CRCl,
                                const VecXd_list &CRTl,
                                Mat42_list &CPs,
                                MatX3d_list &CLCs,
                                VecXd_list &CLTs,
                                MatX3d_list &CRCs,
                                VecXd_list &CRTs);

    void discretize_diffcurv(const Mat42_list &CPs,
                            const MatX3d_list &CLCs,
                            const VecXd_list &CLTs,
                            const MatX3d_list &CRCs,
                            const VecXd_list &CRTs,
                            const Eigen::VectorXi &nel,
                            const VecXd_list &xel,
                            const VecXd_list &xecl,
                            Eigen::MatrixX2d &P,
                            Eigen::MatrixX2d &C,
                            Eigen::MatrixX2d &N,
                            Eigen::MatrixX2i &E,
                            Eigen::VectorXd &L,
                            Eigen::MatrixX3d &u_l,
                            Eigen::MatrixX3d &u_r);

    void discretize_diffcurv(const Mat42_list &CPs,
                            const MatX3d_list &CLCs,
                            const VecXd_list &CLTs,
                            const MatX3d_list &CRCs,
                            const VecXd_list &CRTs,
                            const Eigen::VectorXd &xe,
                            const Eigen::VectorXd &xec,
                            Eigen::MatrixX2d &P,
                            Eigen::MatrixX2d &C,
                            Eigen::MatrixX2d &N,
                            Eigen::MatrixX2i &E,
                            Eigen::VectorXd &L,
                            Eigen::MatrixX3d &u_l,
                            Eigen::MatrixX3d &u_r);

    void discretize_diffcurv(const Mat42_list &CPs,
                            const MatX3d_list &CLCs,
                            const VecXd_list &CLTs,
                            const MatX3d_list &CRCs,
                            const VecXd_list &CRTs,
                            const Eigen::VectorXd &xs,
                            const Eigen::VectorXd &xsc,
                            const Eigen::VectorXd &xg,
                            Eigen::MatrixX2d &P,
                            Eigen::MatrixX2d &C,
                            Eigen::MatrixX2d &Q,
                            Eigen::MatrixX2d &N,
                            Eigen::MatrixX2i &E,
                            Eigen::VectorXd &L,
                            Eigen::MatrixX3d &u_l,
                            Eigen::MatrixX3d &u_r);

    void rediscretize_diffcurv(const Eigen::VectorXd &xs,
                            const Eigen::VectorXd &xsc,
                            const Eigen::VectorXd &xg,
                            const Eigen::VectorXi &subI,
                            const Eigen::VectorXi &origI,
                            const double &sub_t,
                            Mat42_list &CPs,
                            MatX3d_list &CLCs,
                            VecXd_list &CLTs,
                            MatX3d_list &CRCs,
                            VecXd_list &CRTs,
                            Eigen::MatrixX2d &P,
                            Eigen::MatrixX2d &C,
                            Eigen::MatrixX2d &Q,
                            Eigen::MatrixX2d &N,
                            Eigen::MatrixX2i &E,
                            Eigen::VectorXd &L,
                            Eigen::MatrixX3d &u_l,
                            Eigen::MatrixX3d &u_r);

}

#endif