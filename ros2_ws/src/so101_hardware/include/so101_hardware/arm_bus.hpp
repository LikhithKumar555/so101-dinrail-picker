// The whole SO-101 arm on one Feetech bus, in joint units. No ROS inside, so it can be tested
// against a fake bus; the ros2_control plugin (feetech_system.hpp) is a thin layer on top.
//
// One control cycle costs exactly two transactions, independent of what the driver reports:
//   read_state():  ONE SYNC_READ of registers 56..63 (position, velocity, load, voltage,
//                  temperature) for all servos: 14 bytes out + 6 x 14 bytes back = 0.98 ms at 1 Mbps
//   write_goals(): ONE SYNC_WRITE of Goal_Position: 26 bytes = 0.26 ms, no reply
// That leaves most of a 10 ms cycle (100 Hz) free. Six single READs per value would cost
// 6 x 5 registers x ~0.2 ms = 6 ms plus six turn-around delays each.
//
// Two kinds of calls, two rules:
//   control loop (read_state, write_goals): never retry. A lost reply is reported at once and the
//     caller decides; waiting would make the whole 100 Hz loop late.
//   setup (check_servos, set_torque, hold_current_pose): retry up to `setup_attempts` times.
//     Nothing moves yet, a few extra milliseconds cost nothing, and one late reply out of a
//     hundred must not stop the start-up.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "feetech_bus/bus.hpp"
#include "so101_hardware/joint_codec.hpp"

namespace so101_hardware
{

struct ServoState
{
  double position = 0.0;     // rad
  double velocity = 0.0;     // rad/s
  double load = 0.0;         // -1..1: PWM duty the servo applies (sign = joint direction).
                             // At low speed this is roughly the fraction of stall torque.
  double voltage = 0.0;      // V, measured at the servo
  double temperature = 0.0;  // degC, measured inside the servo
  int raw_position = 0;      // encoder steps as read (no clamping, no conversion)
};

class ArmBus
{
public:
  explicit ArmBus(
    std::vector<JointConfig> joints,
    std::chrono::microseconds timeout = std::chrono::milliseconds(10),
    int setup_attempts = 3);

  bool open(const std::string & device, int baud, std::string * error);
  void close() {bus_.close();}
  bool is_open() const {return bus_.is_open();}

  // Read every servo's model number (STS3215 = 777). Says which are missing or wrong.
  bool check_servos(std::string * error);

  // One SYNC_READ for all servos. *error_bits = OR of all servos' status bytes (feetech::err).
  // Control-loop call: no retry.
  feetech::Status read_state(std::vector<ServoState> * state, uint8_t * error_bits = nullptr);

  // Same, retried up to setup_attempts times (setup only).
  feetech::Status read_state_retrying(std::vector<ServoState> * state, std::string * error = nullptr);

  // One SYNC_WRITE of Goal_Position. goals[k] belongs to joints()[k]; angles are clamped to the
  // joint limits. A NaN goal leaves that servo's goal register unchanged.
  feetech::Status write_goals(const std::vector<double> & goals_rad);

  // Torque on/off for all servos, then read Torque_Enable back: SYNC_WRITE has no reply, so
  // reading back is the only way to know that every servo got it. Write + read-back are
  // repeated together if needed: if the write was lost, reading again alone would not help.
  feetech::Status set_torque(bool on, std::string * error = nullptr);

  // Safe start. A servo keeps its last Goal_Position in RAM, also with torque off (someone may
  // have moved the arm by hand since). Switching torque on would make it jump there at full
  // speed. So: read where every joint is, write that RAW reading as the goal (no clamping - if
  // a joint rests beyond its software limit, clamping would also be a jump), read the goals
  // back, and only then switch torque on. *state = the pose that is now being held.
  feetech::Status hold_current_pose(std::vector<ServoState> * state, std::string * error = nullptr);

  const std::vector<JointConfig> & joints() const {return joints_;}
  int setup_retries_used() const {return setup_retries_used_;}   // how often setup had to retry
  const feetech::BusStats & stats() const {return bus_.stats();}

  // "overheat, overload" etc. for a status byte
  static std::string describe_errors(uint8_t error_bits);

private:
  feetech::Status read_back(const feetech::Register & reg, const std::vector<int32_t> & expected,
    std::string * error);
  // SYNC_WRITE values[k] to joints()[k], read back, repeat both until they match (setup only)
  feetech::Status write_verified(const feetech::Register & reg, const std::vector<int32_t> & values,
    std::string * error);

  std::vector<JointConfig> joints_;
  std::vector<uint8_t> ids_;
  int setup_attempts_ = 3;
  int setup_retries_used_ = 0;
  feetech::FeetechBus bus_;
  std::vector<feetech::Bytes> block_;                     // buffers reused from cycle to cycle
  std::vector<std::pair<uint8_t, int32_t>> goals_;
};

}  // namespace so101_hardware
