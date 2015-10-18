#include <eigen/Dense>
#include <iostream>
#include <math.h>
#include <random>
#include <sophus/se3.hpp>
#include <vector>

Eigen::Matrix4d G1;
Eigen::Matrix4d G2;
Eigen::Matrix4d G3;
Eigen::Matrix4d G4;
Eigen::Matrix4d G5;
Eigen::Matrix4d G6;

Eigen::Matrix3d hat(Eigen::Vector3d p)
{
   Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
   m(0,1) = -p(2);
   m(0,2) = p(1);
   m(1,0) = p(2);
   m(1,2) = -p(0);
   m(2,0) = -p(1);
   m(2,1) = p(0);
   return m;   
}

void setup()
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

std::vector<Eigen::Vector2d> residual(const std::vector<Eigen::Vector2d>& A,
                                      const std::vector<Eigen::Vector2d>& B)
{
   std::vector<Eigen::Vector2d> v;
   std::vector<Eigen::Vector2d>::const_iterator i;
   std::vector<Eigen::Vector2d>::const_iterator j;
   for (i = A.begin(), j = B.begin(); i != A.end(); ++i, ++j) {
      v.push_back((*i) - (*j));
   }
   return v;
}

Eigen::Matrix3d toR(double phi, double chi, double psi)
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

Eigen::Matrix4d toT(Eigen::Matrix3d R, Eigen::Vector3d t)
{
   Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
   
   T.block(0, 0, 3, 3) = R;
   T.block(0, 3, 3, 1) = t;
   T(3, 3) = 1.;

   return T;
}

Eigen::Vector2d project(const Eigen::Vector3d& P)
{
   Eigen::Vector2d v;
   v << P(0) / P(2) , P(1) / P(2);
   return v;
}

Eigen::Vector4d homog(const Eigen::Vector3d& P)
{
   Eigen::Vector4d H = Eigen::Vector4d::Zero();
   H(0) = P(0);
   H(1) = P(1);
   H(2) = P(2);
   H(3) = 1.;
   return H;
}

Eigen::VectorXd project(std::vector<Eigen::Vector3d>& P,
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

std::vector<Eigen::Vector4d> homog(const std::vector<Eigen::Vector3d>& P)
{
   std::vector<Eigen::Vector4d> v;
   std::vector<Eigen::Vector3d>::const_iterator i;
   for (i = P.begin(); i != P.end(); ++i) {
      v.push_back(homog(*i));
   }
   return v;
}

// derivative of the pi function
Eigen::Matrix<double, 2, 3> dProj(const Eigen::Vector3d& P)
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

Eigen::Matrix<double, 2, 6> entry(Eigen::Vector3d p)
{
   Eigen::Matrix<double, 3, 6> rMat = Eigen::Matrix<double, 3, 6>::Zero();

   rMat.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
   rMat.block(0, 3, 3, 3) = -hat(p);

   Eigen::Matrix<double, 2, 6> entry = dProj(p) * rMat;

   return entry;   
}

Eigen::MatrixXd computeJ(std::vector<Eigen::Vector3d> P,
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

int main(int argc, char **argv)
{
   // PRNG for noise
   std::default_random_engine generator;
   std::uniform_real_distribution<double> radDist(0.0, 1.0);
   std::uniform_real_distribution<double> dist(0.0, 1.0);

   // create my static generator matrices
   setup();

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

   std::vector<Eigen::Vector3d> v;
   v.push_back(p1);
   v.push_back(p2);
   v.push_back(p3);
   v.push_back(p4);
   v.push_back(p5);
   v.push_back(p6);
   v.push_back(p7);

   // K matrix is 3x4 and is currently set to "identity"
   Eigen::Matrix<double, 3, 4> K = Eigen::Matrix<double, 3, 4>::Zero();
   K(0,0) = 1.;
   K(1,1) = 1.;
   K(2,2) = 1.;

   Eigen::Matrix<double, 4, 4> T = Eigen::Matrix<double, 4, 4>::Identity();

   // T Matrix.  This is what we are going to use to project all our points
   // Then we'll "forget" this matrix and hopefully the optimization process
   // can reconstruct it.
   T = toT(toR(M_PI / 2., 0, 0), Eigen::Vector3d(15, 0, 0));
   T(0,0) = 0.;
   T(1,1) = 0.;
   std::cout << "T:" << std::endl << T << std::endl;
   Eigen::VectorXd v2 = project(v, T, K);

   Eigen::VectorXd obs(v2.size());
   for (size_t i = 0; i < obs.size(); ++i) {
      obs(i) = v2(i) + dist(generator);
   }

   // let's use the points projected using the "real" T as our observations
   obs = v2;

   // let's create an initial guess for T.  it'll be very similar to T,
   // except we'll add some noise the angle we used to create the "real"
   // T
   double noisyAngle = (M_PI / 4.) + radDist(generator);
   Sophus::SE3d se3Guess(toR(noisyAngle, 0, 0), Eigen::Vector3d(10, 10, 1));
   std::cout << "Guess:" << std::endl << se3Guess.matrix() << std::endl;

   for (size_t j = 0; j < 80; j++) {

      // project our points using the T guess
      Eigen::VectorXd pred = project(v, se3Guess.matrix(), K);

      // compute the residual
      Eigen::VectorXd residualV = obs - pred;
      // std::cout << "************ Residual **************" << std::endl;
      // std::cout << residualV << std::endl;
      std::cout << "norm: " << residualV.operatorNorm() << std::endl
                << std::endl;

      if (residualV.operatorNorm() < 1e-12) {
         std::cout << "tolerance reached" << std::endl;
         break;
      }

      Eigen::VectorXd RHS = residualV;

      // compute the Jacobian matrix - and setup the linear system
      Eigen::MatrixXd J = computeJ(v, K, se3Guess.matrix());
      RHS = J.transpose() * RHS;

      Eigen::Matrix<double, 6, 6> M = J.transpose() * J;
      Eigen::FullPivLU<Eigen::MatrixXd> LU(M);
      Eigen::Matrix<double, 6, 1> X = -(LU.solve(RHS));

      // this is something I'm not sure about.  We've solved the
      // system.  now we need to bring it back down from tangent space
      // and do our update.
      se3Guess = se3Guess * Sophus::SE3Group<double>::exp(X);

      // not sure why this left-multiply doesn't work since we left-multiply
      // by the Generators when compuing J.
      // se3Guess = Sophus::SE3Group<double>::exp(X) * se3Guess;

   }

   std::cout << std::endl << "Solution:" << std::endl << se3Guess.matrix() << std::endl;

   return 0;
}
