// ROS 2 node: watches servo telemetry, publishes /diagnostics, and stops the arm on a fault.
//
// Stop strategy ("stop category 2"-like, IEC 60204-1): deactivate the motion controllers so no
// new commands reach the servos; the servos keep their torque and hold the current pose (the
// arm does not fall). The fault is LATCHED: motion is only re-enabled by an explicit reset
// (~/reset service) after the cause is gone.

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "controller_manager_msgs/srv/switch_controller.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "so101_interfaces/msg/servo_telemetry.hpp"
#include "so101_safety/safety_logic.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;
using controller_manager_msgs::srv::SwitchController;
using diagnostic_msgs::msg::DiagnosticArray;
using diagnostic_msgs::msg::DiagnosticStatus;
using so101_interfaces::msg::ServoTelemetry;

namespace so101_safety
{

class SafetyMonitorNode : public rclcpp::Node
{
public:
  SafetyMonitorNode()
  : Node("safety_monitor"), logic_(read_limits())
  {
    controllers_ = declare_parameter<std::vector<std::string>>(
      "controllers_to_stop", {"arm_controller", "gripper_controller"});
    const double rate = declare_parameter<double>("check_rate_hz", 20.0);

    telemetry_sub_ = create_subscription<ServoTelemetry>(
      "servo_telemetry", rclcpp::SensorDataQoS(),
      [this](ServoTelemetry::ConstSharedPtr msg) {on_telemetry(*msg);});

    diag_pub_ = create_publisher<DiagnosticArray>("/diagnostics", 10);
    // Transient-local: a node that starts later still receives the current fault state.
    fault_pub_ = create_publisher<std_msgs::msg::Bool>(
      "~/fault", rclcpp::QoS(1).transient_local().reliable());

    switch_client_ = create_client<SwitchController>("/controller_manager/switch_controller");
    reset_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/reset", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res) {on_reset(*res);});

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate), [this]() {on_timer();});
    publish_fault(false);
    RCLCPP_INFO(get_logger(), "safety monitor running, guarding %zu controllers",
      controllers_.size());
  }

private:
  Limits read_limits()
  {
    Limits l;
    l.load_warn = declare_parameter("load_warn", l.load_warn);
    l.load_fault = declare_parameter("load_fault", l.load_fault);
    l.load_hold_s = declare_parameter("load_hold_s", l.load_hold_s);
    l.temp_warn_c = declare_parameter("temp_warn_c", l.temp_warn_c);
    l.temp_fault_c = declare_parameter("temp_fault_c", l.temp_fault_c);
    l.voltage_min_v = declare_parameter("voltage_min_v", l.voltage_min_v);
    l.voltage_max_v = declare_parameter("voltage_max_v", l.voltage_max_v);
    l.timeout_s = declare_parameter("timeout_s", l.timeout_s);
    l.startup_timeout_s = declare_parameter("startup_timeout_s", l.startup_timeout_s);
    return l;
  }

  void on_telemetry(const ServoTelemetry & msg)
  {
    const size_t n = msg.name.size();
    if (msg.load.size() != n || msg.temperature.size() != n || msg.voltage.size() != n) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "malformed telemetry, ignored");
      return;
    }
    std::vector<ServoReading> readings(n);
    for (size_t i = 0; i < n; ++i) {
      readings[i] = {msg.name[i], msg.load[i], msg.temperature[i], msg.voltage[i]};
    }
    last_status_ = logic_.update(readings, now().seconds());
    handle_level_change();
  }

  void on_timer()
  {
    logic_.check_timeout(now().seconds());
    handle_level_change();
    if (++tick_ % 10 == 0) {  // diagnostics at ~2 Hz is plenty
      publish_diagnostics();
    }
  }

  void handle_level_change()
  {
    const Level level = logic_.level();
    if (level == last_level_) {
      return;
    }
    if (level == Level::FAULT) {
      RCLCPP_ERROR(get_logger(), "FAULT: %s -> stopping motion", logic_.fault_reason().c_str());
      switch_controllers(false);
      publish_fault(true);
    } else if (level == Level::WARN) {
      for (const auto & s : last_status_) {
        if (s.level == Level::WARN) {
          RCLCPP_WARN(get_logger(), "WARN %s: %s", s.name.c_str(), s.message.c_str());
        }
      }
    } else {
      RCLCPP_INFO(get_logger(), "all servos OK");
    }
    last_level_ = level;
  }

  void switch_controllers(bool activate)
  {
    if (!switch_client_->service_is_ready()) {
      RCLCPP_ERROR(get_logger(), "controller_manager/switch_controller not available!");
      return;
    }
    auto req = std::make_shared<SwitchController::Request>();
    (activate ? req->activate_controllers : req->deactivate_controllers) = controllers_;
    req->strictness = SwitchController::Request::BEST_EFFORT;
    switch_client_->async_send_request(
      req, [this, activate](rclcpp::Client<SwitchController>::SharedFuture f) {
        RCLCPP_INFO(get_logger(), "%s controllers: %s", activate ? "activate" : "deactivate",
        f.get()->ok ? "ok" : "FAILED");
      });
  }

  void on_reset(std_srvs::srv::Trigger::Response & res)
  {
    std::string why_not;
    if (!logic_.faulted()) {
      res.success = true;
      res.message = "no latched fault";
      return;
    }
    res.success = logic_.reset(now().seconds(), why_not);
    res.message = res.success ? "fault cleared, motion re-enabled" : "reset refused: " + why_not;
    if (res.success) {
      switch_controllers(true);
      publish_fault(false);
    }
    RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
  }

  void publish_fault(bool fault)
  {
    std_msgs::msg::Bool m;
    m.data = fault;
    fault_pub_->publish(m);
  }

  void publish_diagnostics()
  {
    DiagnosticArray arr;
    arr.header.stamp = now();
    DiagnosticStatus overall;
    overall.name = "so101_safety: overall";
    overall.hardware_id = "so101";
    overall.level = static_cast<uint8_t>(logic_.level());
    overall.message = logic_.faulted() ? "LATCHED FAULT: " + logic_.fault_reason() :
      to_string(logic_.level());
    arr.status.push_back(overall);
    for (const auto & s : last_status_) {
      DiagnosticStatus d;
      d.name = "so101_safety: " + s.name;
      d.hardware_id = "so101/" + s.name;
      d.level = static_cast<uint8_t>(s.level);
      d.message = s.message;
      arr.status.push_back(d);
    }
    diag_pub_->publish(arr);
  }

  SafetyLogic logic_;
  std::vector<std::string> controllers_;
  std::vector<ServoStatus> last_status_;
  Level last_level_ = Level::OK;
  uint64_t tick_ = 0;
  rclcpp::Subscription<ServoTelemetry>::SharedPtr telemetry_sub_;
  rclcpp::Publisher<DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr fault_pub_;
  rclcpp::Client<SwitchController>::SharedPtr switch_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace so101_safety

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<so101_safety::SafetyMonitorNode>());
  rclcpp::shutdown();
  return 0;
}
