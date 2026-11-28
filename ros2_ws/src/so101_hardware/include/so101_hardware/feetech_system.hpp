// ros2_control hardware plugin for the SO-101 on a Feetech STS bus (real arm or virtual bus).
//
// The controller manager runs one loop at update_rate (100 Hz here), always in this order:
//     read()  ->  update() of every active controller  ->  write()
// read() fills the state interfaces (joint position, velocity) from ONE SYNC_READ;
// write() sends the command interfaces (joint position goals) in ONE SYNC_WRITE.
// Both run in the real-time loop: they must not block for long, not allocate much, not throw.
//
// Lifecycle (managed by the controller manager):
//   on_init       parse the <ros2_control> parameters, check the interfaces. No I/O yet.
//   on_configure  open the serial port, PING every servo, start the telemetry publisher
//   on_activate   SAFE START: goal := present position, verify, torque on
//   on_deactivate keep holding the pose (torque stays on: the arm must not fall)
//   on_cleanup    close the port (torque off only if release_torque_on_cleanup = true)
//   on_error      called after read()/write() returned ERROR: the servos keep holding their
//                 last goal on their own - a position servo does not need the host to stand still
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "so101_hardware/arm_bus.hpp"
#include "so101_interfaces/msg/servo_telemetry.hpp"

namespace so101_hardware
{

// Publishes ServoTelemetry from its own thread. The control loop only offers a sample and
// never waits: if the publisher thread holds the lock right now, that sample is skipped.
// (Publishing directly from read() could block on DDS internals for an unknown time.)
class TelemetryPublisher
{
public:
  ~TelemetryPublisher() {stop();}
  void start(const std::string & topic, const std::vector<std::string> & names);
  void stop();
  void offer(const std::vector<ServoState> & state);   // called from the control loop

private:
  void loop();

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<so101_interfaces::msg::ServoTelemetry>::SharedPtr pub_;
  so101_interfaces::msg::ServoTelemetry msg_;
  std::vector<ServoState> latest_;
  bool fresh_ = false;
  bool running_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
};

class FeetechSystem : public hardware_interface::SystemInterface
{
public:
  using CallbackReturn = hardware_interface::CallbackReturn;

  CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams & params)
  override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period)
  override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period)
  override;

private:
  bool count_failure(const char * what, feetech::Status s);   // true = give up (return ERROR)
  void close_bus();

  // parameters from <hardware><param>
  std::string device_;
  int baud_ = 1000000;
  int timeout_ms_ = 10;
  int setup_attempts_ = 3;
  int max_consecutive_errors_ = 5;
  double telemetry_rate_hz_ = 20.0;
  std::string telemetry_topic_ = "/servo_telemetry";
  bool release_torque_on_cleanup_ = false;

  std::vector<JointConfig> joints_;
  std::vector<std::string> position_state_, velocity_state_, position_command_;  // interface names
  std::unique_ptr<ArmBus> arm_;
  std::vector<ServoState> state_;
  std::vector<double> goals_;
  TelemetryPublisher telemetry_;
  int consecutive_errors_ = 0;
  double since_telemetry_s_ = 0.0;
  uint8_t last_error_bits_ = 0;
  bool active_ = false;
};

}  // namespace so101_hardware
