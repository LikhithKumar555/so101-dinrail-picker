#include "so101_hardware/joint_codec.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace so101_hardware
{

namespace
{
// a mod n into 0..n-1, also for negative a (C++ % keeps the sign of a)
int wrap(int a, int n) {return ((a % n) + n) % n;}
}  // namespace

int angle_to_steps(const JointConfig & j, double rad)
{
  const double q = std::clamp(rad, j.min_rad, j.max_rad);
  const long delta = std::lround(q / kRadPerStep);
  return wrap(j.zero_steps + j.direction * static_cast<int>(delta), kStepsPerRev);
}

double steps_to_angle(const JointConfig & j, int steps)
{
  // shortest signed distance from zero_steps, in [-2048, 2047]
  const int d = wrap(steps - j.zero_steps + kStepsPerRev / 2, kStepsPerRev) - kStepsPerRev / 2;
  return j.direction * d * kRadPerStep;
}

double steps_per_s_to_rad_per_s(const JointConfig & j, int steps_per_s)
{
  return j.direction * steps_per_s * kRadPerStep;
}

bool validate(const std::vector<JointConfig> & joints, std::string * error)
{
  std::set<int> ids;
  for (const auto & j : joints) {
    std::string why;
    if (j.id < 1 || j.id > 252) {
      why = "id must be 1..252";
    } else if (!ids.insert(j.id).second) {
      why = "id " + std::to_string(j.id) + " is used twice";
    } else if (j.direction != 1 && j.direction != -1) {
      why = "direction must be 1 or -1";
    } else if (j.zero_steps < 0 || j.zero_steps >= kStepsPerRev) {
      why = "zero_steps must be 0..4095";
    } else if (!(j.min_rad < j.max_rad)) {
      why = "min must be below max";
    } else if (j.max_rad - j.min_rad >= 2.0 * kPi) {
      why = "range must be below one turn (the encoder wraps after 2 pi)";
    }
    if (!why.empty()) {
      if (error) {*error = "joint '" + j.name + "': " + why;}
      return false;
    }
  }
  return true;
}

}  // namespace so101_hardware
