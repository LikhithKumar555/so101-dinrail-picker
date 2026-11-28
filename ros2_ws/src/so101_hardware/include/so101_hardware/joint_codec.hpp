// Joint angle <-> servo encoder steps for one SO-101 joint. No ROS, no I/O: pure arithmetic.
//
// The STS3215 measures its output shaft with a 12-bit magnetic encoder: 4096 steps per turn,
// one step = 2 pi / 4096 = 1.534 mrad = 0.088 deg. A joint angle q [rad] maps to a reading
//
//     steps = zero_steps + direction * q / (2 pi / 4096)        (then wrapped into 0..4095)
//
//   zero_steps  encoder reading when the joint is at q = 0 (the calibration). 2048 in the
//               simulation; on a real arm it comes from a calibration run, because the horn
//               can be mounted at any angle.
//   direction   +1 if the joint angle grows with the encoder count, -1 if the servo is mounted
//               mirrored (same servo, joint turns the other way).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace so101_hardware
{

constexpr int kStepsPerRev = 4096;
constexpr double kPi = 3.14159265358979323846;
constexpr double kRadPerStep = 2.0 * kPi / kStepsPerRev;

struct JointConfig
{
  std::string name;
  uint8_t id = 0;            // servo ID on the bus, 1..252
  int zero_steps = 2048;
  int direction = 1;
  double min_rad = -kPi;     // joint limits; commands are clamped into [min_rad, max_rad]
  double max_rad = kPi;
};

// Goal for a commanded angle: clamped to the joint limits, rounded to the nearest step, 0..4095.
int angle_to_steps(const JointConfig & j, double rad);

// Joint angle for an encoder reading. The difference to zero_steps is taken the short way
// round, so the result is in [-pi, pi): correct for every joint whose range is below one turn.
double steps_to_angle(const JointConfig & j, int steps);

// Present_Velocity is in steps per second.
double steps_per_s_to_rad_per_s(const JointConfig & j, int steps_per_s);

// Checks IDs (1..252, no duplicates), direction (+-1), zero_steps (0..4095) and the limits
// (min < max, range below one turn). Returns false and says why in *error.
bool validate(const std::vector<JointConfig> & joints, std::string * error);

}  // namespace so101_hardware
