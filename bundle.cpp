#include <iostream>
#include <math.h>
#include <random>
#include <sophus/se3.hpp>
#include <vector>
#include "eigen_test.h"

Eigen::Matrix4d G1;
Eigen::Matrix4d G2;
Eigen::Matrix4d G3;
Eigen::Matrix4d G4;
Eigen::Matrix4d G5;
Eigen::Matrix4d G6;

void eigen_test::setup()
{
   G1 = Eigen::Matrix4d::Zero();
   G2 = Eigen::Matrix4d::Zero();
   G3 = Eigen::Matrix4d::Zero();
   G4 = Eigen::Matrix4d::Zero();
   G5 = Eigen::Matrix4d::Zero();
   G6 = Eigen::Matrix4d::Zero();

   G1(0,3) = 1.;
   G2(1,3) = 1.;
   G3(2,3) = 1.;

   G4(1,2) = -1.;
   G4(2,1) = 1.;

   G5(0,2) = 1.;
   G5(2,0) = -1.;

   G6(0,1) = -1.;
   G6(1,0) = 1.;
}

Eigen::Matrix3d eigen_test::toR(double phi, double chi, double psi)
{
   Eigen::Matrix3d R;
   R(0,0) = cos(phi)*cos(chi);
   R(0,1) = cos(phi)*sin(chi)*sin(psi) - sin(phi)*cos(psi);
   R(0,2) = cos(phi)*sin(chi)*cos(psi) + sin(phi)*sin(psi);
   R(1,0) = sin(phi)*cos(chi);
   R(1,1) = sin(phi)*sin(chi)*sin(psi) + cos(phi)*cos(psi);
   R(1,2) = sin(phi)*sin(chi)*cos(psi) - cos(phi)*sin(psi);
   R(2,0) = -sin(chi);
   R(2,1) = cos(chi)*sin(psi);
   R(2,2) = cos(chi)*cos(psi);

   return R;
}

Eigen::Matrix4d eigen_test::toT(Eigen::Matrix3d R, Eigen::Vector3d t)
{
   Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
   
   T.block(0, 0, 3, 3) = R;
   T.block(0, 3, 3, 1) = t;
   T(3, 3) = 1.;

   return T;
}

Eigen::Vector2d eigen_test::project(const Eigen::Vector3d& P)
{
   Eigen::Vector2d v;
   v << P(0) / P(2) , P(1) / P(2);
   return v;
}

Eigen::Vector4d eigen_test::homog(const Eigen::Vector3d& P)
{
   Eigen::Vector4d H = Eigen::Vector4d::Zero();
   H(0) = P(0);
   H(1) = P(1);
   H(2) = P(2);
   H(3) = 1.;
   return H;
}

Eigen::VectorXd eigen_test::project(std::vector<Eigen::Vector3d>& P,
                                    Eigen::Matrix4d T,
                                    Eigen::Matrix<double, 3, 4> K)
{
   Eigen::VectorXd v(2 * P.size());
   std::vector<Eigen::Vector3d>::const_iterator i;
   Eigen::Matrix4d Tinv = T.inverse();
   size_t idx = 0;
   for (i = P.begin(); i != P.end(); ++i) {
      Eigen::Vector2d p = project(K * (Tinv * homog(*i)));
      v(idx++) = p(0);
      v(idx++) = p(1);
   }
   return v;
}

std::vector<Eigen::Vector4d> eigen_test::homog(
    const std::vector<Eigen::Vector3d>& P)
{
   std::vector<Eigen::Vector4d> v;
   std::vector<Eigen::Vector3d>::const_iterator i;
   for (i = P.begin(); i != P.end(); ++i) {
      v.push_back(homog(*i));
   }
   return v;
}

// derivative of the pi function
Eigen::Matrix<double, 2, 3> eigen_test::dProj(const Eigen::Vector3d& P)
{
   Eigen::Matrix<double, 2, 3> dProj = Eigen::Matrix<double, 2, 3>::Zero();

   dProj(0,0) = 1. / P(2);
   dProj(0,1) = 0;
   dProj(0,2) = -P(0) / (P(2)*P(2));
   dProj(1,0) = 0;
   dProj(1,1) = 1. / P(2);
   dProj(1,2) = -P(1) / (P(2)*P(2));

   return dProj;
}

Eigen::MatrixXd eigen_test::computePointBlockJ(std::vector<Eigen::Vector3d> P,
                                               Eigen::Matrix<double, 3, 4> K,
                                               Eigen::Matrix<double, 4, 4> T)
{
   Eigen::MatrixXd J(2 * P.size(), P.size() * 3);
   std::vector<Eigen::Vector3d>::const_iterator i;
   unsigned int idx = 0;

   J = Eigen::MatrixXd::Zero(2 * P.size(), P.size() * 3);

   // The Jacobian I'm trying to compute is for the following function:
   //
   // pi(K * T * Pw) = 
   //
   // dpi/dy * R
   for (i = P.begin(); i != P.end(); ++i) {
      Eigen::Vector3d p = *i;

      Eigen::Matrix4d Tinv = T.inverse();
      Eigen::Vector4d q = Tinv * homog(p);
      Eigen::Matrix3d R = Tinv.block(0, 0, 3, 3);
      Eigen::Matrix<double, 2, 3> block = -dProj(q.block(0,0,3,1)) * R;

      J.block(idx * 2, idx * 3, 2, 3) = block;
      idx++;
   }

   return J;
}

Eigen::MatrixXd eigen_test::computeCameraBlockJ(std::vector<Eigen::Vector3d> P,
                                     Eigen::Matrix<double, 3, 4> K,
                                     Eigen::Matrix<double, 4, 4> T)
{
   Eigen::MatrixXd J(2 * P.size(), 6);
   std::vector<Eigen::Vector3d>::const_iterator i;
   unsigned int idx = 0;

   // The Jacobian I'm trying to compute is for the following function:
   //
   // pi(K * T * Pw)
   //
   // which equals
   //
   // dpi/dy * K * T * Gi * p
   for (i = P.begin(); i != P.end(); ++i) {
      Eigen::Vector3d p = *i;

      Eigen::Vector4d q = T.inverse() * homog(p);
      Eigen::Matrix<double, 2, 1> u1 = -dProj(q.block(0,0,3,1)) * K * G1 * q;
      Eigen::Matrix<double, 2, 1> u2 = -dProj(q.block(0,0,3,1)) * K * G2 * q;
      Eigen::Matrix<double, 2, 1> u3 = -dProj(q.block(0,0,3,1)) * K * G3 * q;
      Eigen::Matrix<double, 2, 1> u4 = -dProj(q.block(0,0,3,1)) * K * G4 * q;
      Eigen::Matrix<double, 2, 1> u5 = -dProj(q.block(0,0,3,1)) * K * G5 * q;
      Eigen::Matrix<double, 2, 1> u6 = -dProj(q.block(0,0,3,1)) * K * G6 * q;

      J.block(idx * 2, 0, 2, 1) = -u1;
      J.block(idx * 2, 1, 2, 1) = -u2;
      J.block(idx * 2, 2, 2, 1) = -u3;
      J.block(idx * 2, 3, 2, 1) = -u4;
      J.block(idx * 2, 4, 2, 1) = -u5;
      J.block(idx * 2, 5, 2, 1) = -u6;
      idx++;
   }

   return J;
}

Eigen::MatrixXd eigen_test::computeJ(std::vector<Eigen::Vector3d> P,
                                     Eigen::Matrix<double, 3, 4> K,
                                     std::vector<Eigen::Matrix<double, 4, 4> > Tlist)
{
   size_t rowIdx = 0;
   size_t colIdx = 0;
   std::vector<Eigen::Matrix<double, 4, 4> >::const_iterator i;

   Eigen::MatrixXd J(2 * P.size() * Tlist.size(),
                     Tlist.size() * 6 + P.size() * 3);

   J = Eigen::MatrixXd::Zero(2 * P.size() * Tlist.size(),
                             Tlist.size() * 6 + P.size() * 3);

   for (i = Tlist.begin(); i != Tlist.end(); ++i) {
      Eigen::Matrix<double, 4, 4> T = *i;
      Eigen::MatrixXd Jblock = computeCameraBlockJ(P, K, T);
      J.block(rowIdx, colIdx, Jblock.rows(), Jblock.cols()) = Jblock;
      colIdx += Jblock.cols();

      Jblock = computePointBlockJ(P, K, T);
      J.block(rowIdx, Tlist.size() * 6, Jblock.rows(), Jblock.cols()) = Jblock;
      rowIdx += Jblock.rows();
   }

   return J;
}

int main(int argc, char **argv)
{
   // PRNG for noise
   std::default_random_engine generator;
   std::uniform_real_distribution<double> radDist(0.0, 1.0);
   std::uniform_real_distribution<double> dist(0.0, 1.0);

   eigen_test BA;

   // create my static generator matrices
   BA.setup();

   // create my points
   Eigen::Vector3d p1;
   p1 << 2, 3, 2;

   Eigen::Vector3d p2;
   p2 << 7, 3, 6;

   Eigen::Vector3d p3;
   p3 << 1, 8, 5;

   Eigen::Vector3d p4;
   p4 << 10, 10, 10;

   Eigen::Vector3d p5;
   p5 << 6, 8, 4;

   Eigen::Vector3d p6;
   p6 << 5, 1, 2;

   Eigen::Vector3d p7;
   p7 << 8, 4, 7;

   Eigen::Vector3d p8;
   p8 << 4, 2, 5;

   Eigen::Vector3d p9;
   p9 << 3, 9, 3;

   Eigen::Vector3d p10;
   p10 << 9, 6, 4;

   std::vector<Eigen::Vector3d> v;
   v.push_back(p1);
   v.push_back(p2);
   v.push_back(p3);
   v.push_back(p4);
   v.push_back(p5);
   v.push_back(p6);
   v.push_back(p7);
   v.push_back(p8);
   v.push_back(p9);
   v.push_back(p10);

   std::vector<Eigen::Vector3d> vGuess;
   for (size_t i = 0; i < v.size(); ++i) {
      vGuess.push_back(Eigen::Vector3d(v[i](0) + 0.01,
                                       v[i](1) + 0.01,
                                       v[i](2) + 0.01));
   }

   // K matrix is 3x4 and is currently set to "identity"
   Eigen::Matrix<double, 3, 4> K = Eigen::Matrix<double, 3, 4>::Zero();
   K(0,0) = 1.;
   K(1,1) = 1.;
   K(2,2) = 1.;

   Eigen::Matrix<double, 4, 4> T1 = Eigen::Matrix<double, 4, 4>::Identity();
   Eigen::Matrix<double, 4, 4> T2 = Eigen::Matrix<double, 4, 4>::Identity();
   Eigen::Matrix<double, 4, 4> T3 = Eigen::Matrix<double, 4, 4>::Identity();

   // T Matrix.
   T1 = BA.toT(BA.toR(M_PI / 2., 0, 0), Eigen::Vector3d(15, 0, 11));
   T1(0,0) = 0.;
   T1(1,1) = 0.;
   std::cout << "T1:" << std::endl << T1 << std::endl;
   Eigen::VectorXd v1 = BA.project(v, T1, K);

   T2 = BA.toT(BA.toR(M_PI * .4, 0, 0), Eigen::Vector3d(30, 12, 19));
   // T2(0,0) = 0.;
   // T2(1,1) = 0.;
   std::cout << "T2:" << std::endl << T2 << std::endl;
   Eigen::VectorXd v2 = BA.project(v, T2, K);

   T3 = BA.toT(BA.toR(M_PI * .3, 0, 0), Eigen::Vector3d(45, 14, 17));
   // T3(0,0) = 0.;
   // T3(1,1) = 0.;
   std::cout << "T3:" << std::endl << T3 << std::endl;
   Eigen::VectorXd v3 = BA.project(v, T3, K);

   Eigen::VectorXd v4(v1.rows() + v2.rows() + v3.rows());
   v4 << v1, v2, v3;

   Eigen::VectorXd obs(v4.size());
   obs = v4;
   std::cout << "Obs:" << std::endl << obs << std::endl;

   Sophus::SE3d se3Guess1(BA.toR((M_PI / 2.), 0, 0),
                          Eigen::Vector3d(15, 0, 11));

   Sophus::SE3d se3Guess2(BA.toR((M_PI * .4), 0, 0),
                          Eigen::Vector3d(30, 12, 19));

   Sophus::SE3d se3Guess3(BA.toR((M_PI * .3), 0, 0),
                          Eigen::Vector3d(45, 14, 17));

   std::vector<Eigen::Matrix<double, 4, 4> > Tlist;
   Tlist.push_back(se3Guess1.matrix());
   Tlist.push_back(se3Guess2.matrix());
   Tlist.push_back(se3Guess3.matrix());

   for (size_t j = 0; j < 1000; j++) {

      // project our points using the 3D guess for the points
      Eigen::VectorXd pred1 = BA.project(vGuess, se3Guess1.matrix(), K);
      Eigen::VectorXd pred2 = BA.project(vGuess, se3Guess2.matrix(), K);
      Eigen::VectorXd pred3 = BA.project(vGuess, se3Guess3.matrix(), K);
      Eigen::VectorXd pred4(pred1.rows() + pred2.rows() + pred3.rows());
      pred4 << pred1, pred2, pred3;

      // compute the residual
      Eigen::VectorXd residualV = obs - pred4;
      // std::cout << "************ Obs **************" << std::endl;
      // std::cout << obs << std::endl;
      // std::cout << "************ Pred **************" << std::endl;
      // std::cout <<  pred4 << std::endl;
      // std::cout << "************ Residual **************" << std::endl;
      // std::cout << residualV << std::endl;
      std::cout << "norm: " << residualV.operatorNorm() << std::endl
                << std::endl;

      if (fabs(residualV.operatorNorm()) < 1e-12) {
         std::cout << "tolerance reached in " << j+1 << " iterations"
                   << std::endl;
         break;
      }

      Eigen::VectorXd RHS = residualV;

      // compute the Jacobian matrix - and setup the linear system
      Eigen::MatrixXd J = BA.computeJ(vGuess, K, Tlist);
      RHS = J.transpose() * RHS;

      Eigen::MatrixXd M = J.transpose() * J;
      Eigen::FullPivLU<Eigen::MatrixXd> LU(M);
      Eigen::VectorXd X = -(LU.solve(RHS));

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);

      double cond = svd.singularValues()(0) /
         svd.singularValues()(svd.singularValues().size()-1);

      std::cout << "Cond#: " << cond << std::endl;

      se3Guess1 = se3Guess1 *Sophus::SE3Group<double>::exp(X.block(0,0,6,1));
      se3Guess2 = se3Guess2 *Sophus::SE3Group<double>::exp(X.block(6,0,6,1));
      se3Guess3 = se3Guess3 *Sophus::SE3Group<double>::exp(X.block(12,0,6,1));
      //std::cout << "T1Guess:" << std::endl << se3Guess1.matrix() << std::endl;
      //std::cout << "T2Guess:" << std::endl << se3Guess2.matrix() << std::endl;
      //std::cout << "T3Guess:" << std::endl << se3Guess3.matrix() << std::endl;

      size_t offset = Tlist.size() * 6;
      for (size_t i = 0; i < vGuess.size(); ++i) {
         Eigen::Vector3d p = X.block(offset + (i * 3), 0, 3, 1);
         vGuess[i] += p;
         // std::cout << i << ":" << std::endl << vGuess[i] << std::endl;
      }
   }

   std::cout << "T1Guess:" << std::endl << se3Guess1.matrix() << std::endl;
   std::cout << "T2Guess:" << std::endl << se3Guess2.matrix() << std::endl;
   std::cout << "T3Guess:" << std::endl << se3Guess3.matrix() << std::endl;

   std::cout << "Reconstructed 3D points." << std::endl;
   for (size_t i = 0; i < vGuess.size(); ++i) {
      std::cout << vGuess[i] << std::endl << std::endl;
   }

   std::cout << "Reprojected 3D points - compare to Obs." << std::endl;
   for (size_t i = 0; i < vGuess.size(); ++i) {
      std::cout << BA.project(K * (se3Guess1.matrix().inverse() * BA.homog(vGuess[i]))) << std::endl;
   }

   for (size_t i = 0; i < vGuess.size(); ++i) {
      std::cout << BA.project(K * (se3Guess2.matrix().inverse() * BA.homog(vGuess[i]))) << std::endl;
   }

   for (size_t i = 0; i < vGuess.size(); ++i) {
      std::cout << BA.project(K * (se3Guess3.matrix().inverse() * BA.homog(vGuess[i]))) << std::endl;
   }

   return 0;
}
