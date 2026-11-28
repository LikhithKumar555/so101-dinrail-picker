#include "so101_hardware/arm_bus.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace so101_hardware
{

namespace reg = feetech::reg;
using feetech::Status;

namespace
{
// Present_Position (56) .. Present_Temperature (63): one contiguous block in the control table
constexpr uint8_t kBlockStart = reg::kPresentPosition.address;
constexpr uint8_t kBlockLength =
  reg::kPresentTemperature.address + reg::kPresentTemperature.size - kBlockStart;
static_assert(kBlockLength == 8, "STS3215 status block is 8 bytes");
}  // namespace

ArmBus::ArmBus(
  std::vector<JointConfig> joints, std::chrono::microseconds timeout, int setup_attempts)
: joints_(std::move(joints)), setup_attempts_(std::max(1, setup_attempts)), bus_(timeout)
{
  for (const auto & j : joints_) {
    ids_.push_back(j.id);
  }
}

bool ArmBus::open(const std::string & device, int baud, std::string * error)
{
  return bus_.open(device, baud, error);
}

bool ArmBus::check_servos(std::string * error)
{
  std::string problems;
  for (const auto & j : joints_) {
    int32_t model = 0;
    Status s = Status::kTimeout;
    for (int attempt = 0; attempt < setup_attempts_; ++attempt) {
      s = bus_.read(j.id, reg::kModelNumber, &model);
      if (s == Status::kOk) {break;}
      ++setup_retries_used_;
    }
    if (s != Status::kOk) {
      problems += " " + j.name + " (id " + std::to_string(j.id) + "): " + feetech::to_string(s) + ";";
    } else if (model != feetech::kSts3215ModelNumber) {
      problems += " " + j.name + " (id " + std::to_string(j.id) + "): model " +
        std::to_string(model) + " is not an STS3215;";
    }
  }
  if (!problems.empty() && error) {*error = "servo check failed:" + problems;}
  return problems.empty();
}

Status ArmBus::read_state(std::vector<ServoState> * state, uint8_t * error_bits)
{
  const Status s = bus_.sync_read_block(kBlockStart, kBlockLength, ids_, &block_, error_bits);
  if (s != Status::kOk) {return s;}
  state->resize(joints_.size());
  for (std::size_t k = 0; k < joints_.size(); ++k) {
    const JointConfig & j = joints_[k];
    const uint8_t * d = block_[k].data();
    auto field = [d](const feetech::Register & r) {
        return feetech::decode_value(r, d + (r.address - kBlockStart));
      };
    ServoState & out = (*state)[k];
    out.raw_position = field(reg::kPresentPosition);
    out.position = steps_to_angle(j, out.raw_position);
    out.velocity = steps_per_s_to_rad_per_s(j, field(reg::kPresentVelocity));
    out.load = j.direction * field(reg::kPresentLoad) / 1000.0;     // register unit: 0.1 % duty
    out.voltage = field(reg::kPresentVoltage) * 0.1;                // register unit: 0.1 V
    out.temperature = field(reg::kPresentTemperature);              // register unit: 1 degC
  }
  return Status::kOk;
}

Status ArmBus::write_goals(const std::vector<double> & goals_rad)
{
  goals_.clear();
  for (std::size_t k = 0; k < joints_.size() && k < goals_rad.size(); ++k) {
    if (!std::isnan(goals_rad[k])) {
      goals_.emplace_back(joints_[k].id, angle_to_steps(joints_[k], goals_rad[k]));
    }
  }
  if (goals_.empty()) {return Status::kOk;}
  return bus_.sync_write(reg::kGoalPosition, goals_);
}

Status ArmBus::read_back(
  const feetech::Register & r, const std::vector<int32_t> & expected, std::string * error)
{
  std::vector<int32_t> got;
  const Status s = bus_.sync_read(r, ids_, &got);
  if (s != Status::kOk) {
    if (error) {*error = std::string("reading back ") + r.name + ": " + feetech::to_string(s);}
    return s;
  }
  std::string wrong;
  for (std::size_t k = 0; k < ids_.size(); ++k) {
    if (got[k] != expected[k]) {
      wrong += " " + joints_[k].name + " has " + std::to_string(got[k]) + " instead of " +
        std::to_string(expected[k]) + ";";
    }
  }
  if (!wrong.empty()) {
    if (error) {*error = std::string(r.name) + " did not arrive:" + wrong;}
    return Status::kBadReply;
  }
  return Status::kOk;
}

Status ArmBus::read_state_retrying(std::vector<ServoState> * state, std::string * error)
{
  Status s = Status::kTimeout;
  for (int attempt = 0; attempt < setup_attempts_; ++attempt) {
    s = read_state(state);
    if (s == Status::kOk) {return s;}
    ++setup_retries_used_;
  }
  if (error) {*error = std::string("reading the state: ") + feetech::to_string(s);}
  return s;
}

Status ArmBus::write_verified(
  const feetech::Register & r, const std::vector<int32_t> & values, std::string * error)
{
  std::vector<std::pair<uint8_t, int32_t>> packet;
  for (std::size_t k = 0; k < ids_.size(); ++k) {
    packet.emplace_back(ids_[k], values[k]);
  }
  Status s = Status::kTimeout;
  for (int attempt = 0; attempt < setup_attempts_; ++attempt) {
    if (attempt > 0) {++setup_retries_used_;}
    s = bus_.sync_write(r, packet);
    if (s != Status::kOk) {   // the port itself failed: retrying will not help
      if (error) {*error = std::string("writing ") + r.name + ": " + feetech::to_string(s);}
      return s;
    }
    s = read_back(r, values, error);
    if (s == Status::kOk) {return s;}
  }
  return s;                   // *error describes the last attempt
}

Status ArmBus::set_torque(bool on, std::string * error)
{
  return write_verified(reg::kTorqueEnable, std::vector<int32_t>(ids_.size(), on ? 1 : 0), error);
}

Status ArmBus::hold_current_pose(std::vector<ServoState> * state, std::string * error)
{
  Status s = read_state_retrying(state, error);
  if (s != Status::kOk) {return s;}
  std::vector<int32_t> goals;
  for (const auto & st : *state) {
    goals.push_back(st.raw_position);   // the raw reading: no clamping, no conversion
  }
  s = write_verified(reg::kGoalPosition, goals, error);
  if (s != Status::kOk) {return s;}     // goals not confirmed: torque stays OFF
  return set_torque(true, error);
}

std::string ArmBus::describe_errors(uint8_t bits)
{
  namespace err = feetech::err;
  static const std::pair<uint8_t, const char *> names[] = {
    {err::kVoltage, "voltage"}, {err::kAngle, "angle limit"}, {err::kOverheat, "overheat"},
    {err::kOverCurrent, "overcurrent"}, {err::kChecksum, "checksum"}, {err::kOverload, "overload"},
    {err::kInstruction, "instruction"}};
  std::string out;
  for (const auto & [bit, name] : names) {
    if (bits & bit) {
      out += (out.empty() ? "" : ", ") + std::string(name);
    }
  }
  return out.empty() ? "none" : out;
}

}  // namespace so101_hardware
