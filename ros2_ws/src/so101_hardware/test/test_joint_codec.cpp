#include <gtest/gtest.h>

#include <cmath>

#include "so101_hardware/joint_codec.hpp"

using so101_hardware::JointConfig;
using so101_hardware::kPi;
using so101_hardware::kRadPerStep;

namespace
{
JointConfig joint(int zero = 2048, int direction = 1, double lo = -2.0, double hi = 2.0)
{
  JointConfig j;
  j.name = "test";
  j.id = 1;
  j.zero_steps = zero;
  j.direction = direction;
  j.min_rad = lo;
  j.max_rad = hi;
  return j;
}
}  // namespace

TEST(JointCodec, ZeroAndQuarterTurn)
{
  const auto j = joint();
  EXPECT_EQ(so101_hardware::angle_to_steps(j, 0.0), 2048);
  EXPECT_EQ(so101_hardware::angle_to_steps(j, kPi / 2), 2048 + 1024);
  EXPECT_EQ(so101_hardware::angle_to_steps(j, -kPi / 2), 2048 - 1024);
  EXPECT_NEAR(so101_hardware::steps_to_angle(j, 3072), kPi / 2, 1e-12);
}

TEST(JointCodec, MirroredServoTurnsTheOtherWay)
{
  const auto j = joint(2048, -1);
  EXPECT_EQ(so101_hardware::angle_to_steps(j, kPi / 2), 2048 - 1024);
  EXPECT_NEAR(so101_hardware::steps_to_angle(j, 1024), kPi / 2, 1e-12);
  EXPECT_NEAR(so101_hardware::steps_per_s_to_rad_per_s(j, 100), -100 * kRadPerStep, 1e-12);
}

TEST(JointCodec, RoundTripErrorIsAtMostHalfAStep)
{
  const auto j = joint(1500, 1);
  double worst = 0.0;
  for (double q = -2.0; q <= 2.0; q += 0.0007) {
    const double back = so101_hardware::steps_to_angle(j, so101_hardware::angle_to_steps(j, q));
    worst = std::max(worst, std::abs(back - q));
  }
  EXPECT_LE(worst, 0.5 * kRadPerStep + 1e-12);   // 0.77 mrad = 0.044 deg
}

TEST(JointCodec, CommandsAreClampedToTheLimits)
{
  const auto j = joint(2048, 1, -1.0, 1.0);
  EXPECT_EQ(so101_hardware::angle_to_steps(j, 5.0), so101_hardware::angle_to_steps(j, 1.0));
  EXPECT_EQ(so101_hardware::angle_to_steps(j, -5.0), so101_hardware::angle_to_steps(j, -1.0));
}

TEST(JointCodec, WrapsAcrossTheEncoderZero)
{
  // calibration close to the encoder's 0/4095 seam: a joint at -0.5 rad reads ~3770
  const auto j = joint(100, 1);
  const int s = so101_hardware::angle_to_steps(j, -0.5);
  EXPECT_EQ(s, 100 - 326 + 4096);
  EXPECT_NEAR(so101_hardware::steps_to_angle(j, s), -0.5, 0.5 * kRadPerStep);
  EXPECT_NEAR(so101_hardware::steps_to_angle(j, 400), 300 * kRadPerStep, 1e-12);
}

TEST(JointCodec, ValidateFindsBadConfigurations)
{
  std::string why;
  auto a = joint();
  auto b = joint();
  EXPECT_FALSE(so101_hardware::validate({a, b}, &why));          // same id twice
  EXPECT_NE(why.find("twice"), std::string::npos);
  b.id = 2;
  EXPECT_TRUE(so101_hardware::validate({a, b}, &why));
  b.direction = 0;
  EXPECT_FALSE(so101_hardware::validate({a, b}, &why));
  b = joint(2048, 1, -3.2, 3.2);
  b.id = 2;
  EXPECT_FALSE(so101_hardware::validate({a, b}, &why));          // more than one turn
}
