
#ifndef EIGEN_TEST_H
#define EIGEN_TEST_H

#include <eigen/Dense>

class eigen_test
{
  public:
   eigen_test() {}
   virtual ~eigen_test() {}

   void setup();

   Eigen::Matrix3d toR(double phi, double chi, double psi);

   Eigen::Matrix4d toT(Eigen::Matrix3d R, Eigen::Vector3d t);

   Eigen::Vector2d project(const Eigen::Vector3d& P);

   Eigen::Vector4d homog(const Eigen::Vector3d& P);

   Eigen::VectorXd project(std::vector<Eigen::Vector3d>& P,
                           Eigen::Matrix4d T,
                           Eigen::Matrix<double, 3, 4> K);

   std::vector<Eigen::Vector4d> homog(const std::vector<Eigen::Vector3d>& P);

   Eigen::Matrix<double, 2, 3> dProj(const Eigen::Vector3d& P);

   Eigen::MatrixXd computeJ(std::vector<Eigen::Vector3d> P,
                            Eigen::Matrix<double, 3, 4> K,
                            std::vector<Eigen::Matrix<double, 4, 4> > Tlist);

   Eigen::MatrixXd computeJ(std::vector<Eigen::Vector3d> P,
                            Eigen::Matrix<double, 3, 4> K,
                            Eigen::Matrix<double, 4, 4> T);

   Eigen::MatrixXd computeCameraBlockJ(std::vector<Eigen::Vector3d> P,
                                       Eigen::Matrix<double, 3, 4> K,
                                       Eigen::Matrix<double, 4, 4> T);
};

#endif
