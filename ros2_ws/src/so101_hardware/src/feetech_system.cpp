#include "so101_hardware/feetech_system.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace so101_hardware
{

using hardware_interface::return_type;
using CallbackReturn = hardware_interface::CallbackReturn;

// ------------------------------------------------------------------ telemetry publisher

void TelemetryPublisher::start(const std::string & topic, const std::vector<std::string> & names)
{
  stop();
  rclcpp::NodeOptions options;
  // not affected by the command-line arguments of ros2_control_node (node name remaps etc.)
  options.use_global_arguments(false).start_parameter_services(false)
  .start_parameter_event_publisher(false);
  node_ = std::make_shared<rclcpp::Node>("so101_hardware_telemetry", options);
  // best effort, like the safety monitor's subscription: an old sample is worth nothing
  pub_ = node_->create_publisher<so101_interfaces::msg::ServoTelemetry>(
    topic, rclcpp::SensorDataQoS());
  msg_.name = names;
  msg_.load.resize(names.size());
  msg_.temperature.resize(names.size());
  msg_.voltage.resize(names.size());
  {
    std::lock_guard<std::mutex> lk(mutex_);
    running_ = true;
    fresh_ = false;
  }
  thread_ = std::thread([this] {loop();});
}

void TelemetryPublisher::stop()
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    running_ = false;
  }
  cv_.notify_all();
  if (thread_.joinable()) {thread_.join();}
  pub_.reset();
  node_.reset();
}

void TelemetryPublisher::offer(const std::vector<ServoState> & state)
{
  std::unique_lock<std::mutex> lk(mutex_, std::try_to_lock);
  if (!lk.owns_lock() || !running_) {return;}   // publisher busy: skip, never wait
  latest_ = state;                              // same size every time: no reallocation
  fresh_ = true;
  lk.unlock();
  cv_.notify_one();
}

void TelemetryPublisher::loop()
{
  std::unique_lock<std::mutex> lk(mutex_);
  while (true) {
    cv_.wait(lk, [this] {return fresh_ || !running_;});
    if (!running_) {return;}
    fresh_ = false;
    for (std::size_t k = 0; k < latest_.size() && k < msg_.name.size(); ++k) {
      msg_.load[k] = latest_[k].load;
      msg_.temperature[k] = latest_[k].temperature;
      msg_.voltage[k] = latest_[k].voltage;
    }
    lk.unlock();                                // publish without holding the lock
    msg_.header.stamp = node_->now();
    pub_->publish(msg_);
    lk.lock();
  }
}

// ------------------------------------------------------------------ parameter parsing

namespace
{
std::string param(
  const std::unordered_map<std::string, std::string> & params, const std::string & key,
  const std::string & fallback)
{
  const auto it = params.find(key);
  return it == params.end() ? fallback : it->second;
}

bool to_bool(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {return std::tolower(c);});
  if (s == "true" || s == "1") {return true;}
  if (s == "false" || s == "0") {return false;}
  throw std::invalid_argument("'" + s + "' is not true/false");
}

double limit_or(const std::string & s, double fallback)
{
  return s.empty() ? fallback : std::stod(s);
}
}  // namespace

// ------------------------------------------------------------------ lifecycle

CallbackReturn FeetechSystem::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  const auto & hw = info_.hardware_parameters;
  try {
    device_ = param(hw, "device", "/dev/ttyACM0");
    baud_ = std::stoi(param(hw, "baud", "1000000"));
    timeout_ms_ = std::stoi(param(hw, "timeout_ms", "10"));
    setup_attempts_ = std::stoi(param(hw, "setup_attempts", "3"));
    max_consecutive_errors_ = std::stoi(param(hw, "max_consecutive_errors", "5"));
    telemetry_rate_hz_ = std::stod(param(hw, "telemetry_rate_hz", "20"));
    telemetry_topic_ = param(hw, "telemetry_topic", "/servo_telemetry");
    release_torque_on_cleanup_ = to_bool(param(hw, "release_torque_on_cleanup", "false"));

    joints_.clear();
    position_state_.clear();
    velocity_state_.clear();
    position_command_.clear();
    for (const auto & j : info_.joints) {
      // exactly one command interface (position) and the states position + velocity
      if (j.command_interfaces.size() != 1 ||
        j.command_interfaces[0].name != hardware_interface::HW_IF_POSITION)
      {
        RCLCPP_ERROR(get_logger(), "joint '%s': needs exactly one command interface 'position'",
          j.name.c_str());
        return CallbackReturn::ERROR;
      }
      std::vector<std::string> states;
      for (const auto & s : j.state_interfaces) {states.push_back(s.name);}
      std::sort(states.begin(), states.end());
      if (states != std::vector<std::string>{"position", "velocity"}) {
        RCLCPP_ERROR(get_logger(), "joint '%s': state interfaces must be 'position' and 'velocity'",
          j.name.c_str());
        return CallbackReturn::ERROR;
      }
      if (!j.parameters.count("id")) {
        RCLCPP_ERROR(get_logger(), "joint '%s': <param name=\"id\"> (servo ID) is missing",
          j.name.c_str());
        return CallbackReturn::ERROR;
      }
      JointConfig c;
      c.name = j.name;
      c.id = static_cast<uint8_t>(std::stoi(j.parameters.at("id")));
      c.zero_steps = std::stoi(param(j.parameters, "zero_steps", "2048"));
      c.direction = std::stoi(param(j.parameters, "direction", "1"));
      c.min_rad = limit_or(j.command_interfaces[0].min, -kPi);
      c.max_rad = limit_or(j.command_interfaces[0].max, kPi);
      joints_.push_back(c);
      position_state_.push_back(j.name + "/" + hardware_interface::HW_IF_POSITION);
      velocity_state_.push_back(j.name + "/" + hardware_interface::HW_IF_VELOCITY);
      position_command_.push_back(j.name + "/" + hardware_interface::HW_IF_POSITION);
    }
  } catch (const std::exception & e) {   // stoi/stod/to_bool on a malformed value
    RCLCPP_ERROR(get_logger(), "bad parameter: %s", e.what());
    return CallbackReturn::ERROR;
  }
  std::string why;
  if (!validate(joints_, &why)) {
    RCLCPP_ERROR(get_logger(), "%s", why.c_str());
    return CallbackReturn::ERROR;
  }
  if (timeout_ms_ < 1 || max_consecutive_errors_ < 1 || setup_attempts_ < 1) {
    RCLCPP_ERROR(get_logger(), "timeout_ms, max_consecutive_errors, setup_attempts must be >= 1");
    return CallbackReturn::ERROR;
  }
  state_.assign(joints_.size(), ServoState{});
  goals_.assign(joints_.size(), std::nan(""));
  RCLCPP_INFO(get_logger(), "%zu Feetech joints on %s @ %d baud, timeout %d ms",
    joints_.size(), device_.c_str(), baud_, timeout_ms_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn FeetechSystem::on_configure(const rclcpp_lifecycle::State &)
{
  arm_ = std::make_unique<ArmBus>(
    joints_, std::chrono::milliseconds(timeout_ms_), setup_attempts_);
  std::string error;
  if (!arm_->open(device_, baud_, &error)) {
    RCLCPP_ERROR(get_logger(), "cannot open %s: %s", device_.c_str(), error.c_str());
    arm_.reset();
    return CallbackReturn::ERROR;
  }
  if (!arm_->check_servos(&error)) {
    RCLCPP_ERROR(get_logger(), "%s", error.c_str());
    close_bus();
    return CallbackReturn::ERROR;
  }
  if (arm_->read_state_retrying(&state_, &error) != feetech::Status::kOk) {
    RCLCPP_ERROR(get_logger(), "first %s", error.c_str());
    close_bus();
    return CallbackReturn::ERROR;
  }
  for (std::size_t k = 0; k < joints_.size(); ++k) {
    set_state(position_state_[k], state_[k].position);
    set_state(velocity_state_[k], 0.0);
    RCLCPP_INFO(get_logger(), "  %-14s id %d  %+7.3f rad  %4.1f V  %2.0f degC",
      joints_[k].name.c_str(), joints_[k].id, state_[k].position, state_[k].voltage,
      state_[k].temperature);
  }
  if (telemetry_rate_hz_ > 0.0) {
    std::vector<std::string> names;
    for (const auto & j : joints_) {names.push_back(j.name);}
    telemetry_.start(telemetry_topic_, names);
  }
  consecutive_errors_ = 0;
  RCLCPP_INFO(get_logger(), "configured: all %zu servos answer (torque still off, %d setup "
    "retries)", joints_.size(), arm_->setup_retries_used());
  return CallbackReturn::SUCCESS;
}

CallbackReturn FeetechSystem::on_activate(const rclcpp_lifecycle::State &)
{
  std::string error;
  if (arm_->hold_current_pose(&state_, &error) != feetech::Status::kOk) {
    RCLCPP_ERROR(get_logger(), "safe start failed, torque NOT enabled: %s", error.c_str());
    return CallbackReturn::ERROR;
  }
  for (std::size_t k = 0; k < joints_.size(); ++k) {
    set_state(position_state_[k], state_[k].position);
    set_state(velocity_state_[k], 0.0);
    set_command(position_command_[k], state_[k].position);   // first command = stay here
    goals_[k] = state_[k].position;
    const JointConfig & j = joints_[k];
    if (state_[k].position < j.min_rad - 0.02 || state_[k].position > j.max_rad + 0.02) {
      RCLCPP_WARN(get_logger(), "%s is at %.3f rad, outside its limits [%.3f, %.3f]: the first "
        "command will move it to the limit", j.name.c_str(), state_[k].position, j.min_rad,
        j.max_rad);
    }
  }
  consecutive_errors_ = 0;
  since_telemetry_s_ = 0.0;
  active_ = true;
  RCLCPP_INFO(get_logger(), "active: torque on, holding the current pose (%d setup retries so "
    "far)", arm_->setup_retries_used());
  return CallbackReturn::SUCCESS;
}

CallbackReturn FeetechSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  active_ = false;
  RCLCPP_INFO(get_logger(), "inactive: no new goals; the servos keep holding the last one");
  return CallbackReturn::SUCCESS;
}

void FeetechSystem::close_bus()
{
  telemetry_.stop();
  if (arm_) {arm_->close();}
  arm_.reset();
}

CallbackReturn FeetechSystem::on_cleanup(const rclcpp_lifecycle::State &)
{
  active_ = false;
  if (arm_ && release_torque_on_cleanup_) {
    RCLCPP_WARN(get_logger(), "releasing torque: the arm will sag under gravity");
    std::string error;
    if (arm_->set_torque(false, &error) != feetech::Status::kOk) {
      RCLCPP_ERROR(get_logger(), "torque off failed: %s", error.c_str());
    }
  }
  close_bus();
  return CallbackReturn::SUCCESS;
}

CallbackReturn FeetechSystem::on_shutdown(const rclcpp_lifecycle::State & previous_state)
{
  return on_cleanup(previous_state);
}

CallbackReturn FeetechSystem::on_error(const rclcpp_lifecycle::State &)
{
  active_ = false;
  RCLCPP_ERROR(get_logger(), "hardware error: bus closed. The servos still hold their last "
    "goal (if they have power). Fix the cause, then configure + activate again.");
  close_bus();
  return CallbackReturn::SUCCESS;   // -> UNCONFIGURED: can be configured again
}

// ------------------------------------------------------------------ control loop

bool FeetechSystem::count_failure(const char * what, feetech::Status s)
{
  ++consecutive_errors_;
  RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s: %s (%d in a row, limit %d)",
    what, feetech::to_string(s), consecutive_errors_, max_consecutive_errors_);
  if (consecutive_errors_ >= max_consecutive_errors_) {
    RCLCPP_ERROR(get_logger(), "%d bus errors in a row - giving up", consecutive_errors_);
    return true;
  }
  return false;
}

return_type FeetechSystem::read(const rclcpp::Time &, const rclcpp::Duration & period)
{
  if (!arm_) {return return_type::ERROR;}
  uint8_t bits = 0;
  const feetech::Status s = arm_->read_state(&state_, &bits);
  if (s != feetech::Status::kOk) {
    // keep the last state for this cycle; a single lost reply is normal on a serial bus
    return count_failure("read", s) ? return_type::ERROR : return_type::OK;
  }
  consecutive_errors_ = 0;
  for (std::size_t k = 0; k < joints_.size(); ++k) {
    set_state(position_state_[k], state_[k].position);
    set_state(velocity_state_[k], state_[k].velocity);
  }
  if (bits != last_error_bits_) {        // log changes only, not every cycle
    if (bits) {
      RCLCPP_WARN(get_logger(), "servo status: %s", ArmBus::describe_errors(bits).c_str());
    } else {
      RCLCPP_INFO(get_logger(), "servo status: ok again");
    }
    last_error_bits_ = bits;
  }
  if (telemetry_rate_hz_ > 0.0) {
    since_telemetry_s_ += period.seconds();
    if (since_telemetry_s_ >= 1.0 / telemetry_rate_hz_) {
      telemetry_.offer(state_);
      since_telemetry_s_ = 0.0;
    }
  }
  return return_type::OK;
}

return_type FeetechSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!active_ || !arm_) {return return_type::OK;}
  for (std::size_t k = 0; k < joints_.size(); ++k) {
    goals_[k] = get_command(position_command_[k]);
  }
  const feetech::Status s = arm_->write_goals(goals_);
  if (s != feetech::Status::kOk) {
    return count_failure("write", s) ? return_type::ERROR : return_type::OK;
  }
  return return_type::OK;
}

}  // namespace so101_hardware

PLUGINLIB_EXPORT_CLASS(so101_hardware::FeetechSystem, hardware_interface::SystemInterface)
