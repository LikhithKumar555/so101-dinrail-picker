// Pure C++ safety rules for the SO-101 servos (no ROS inside, so it can be unit-tested).
#pragma once

#include <map>
#include <string>
#include <vector>

namespace so101_safety
{

enum class Level { OK = 0, WARN = 1, FAULT = 2 };

const char * to_string(Level level);

struct Limits
{
  double load_warn = 0.60;       // |load| (fraction of max torque) that triggers a warning
  double load_fault = 0.85;      // |load| that is a fault ...
  double load_hold_s = 0.5;      // ... if it lasts this long (short spikes while accelerating are normal)
  double temp_warn_c = 55.0;     // degC
  double temp_fault_c = 65.0;    // degC
  double voltage_min_v = 6.0;    // below: brown-out, servos may reset
  double voltage_max_v = 13.0;   // above: wrong power supply
  double timeout_s = 0.5;        // no telemetry for this long = fault (we are blind)
  double startup_timeout_s = 5.0;  // grace period for the first message after start-up
};

struct ServoReading
{
  std::string name;
  double load = 0.0;
  double temperature_c = 0.0;
  double voltage_v = 0.0;
};

struct ServoStatus
{
  std::string name;
  Level level = Level::OK;
  std::string message;
};

class SafetyLogic
{
public:
  explicit SafetyLogic(const Limits & limits);

  // Evaluate one telemetry sample received at time t [s]. Returns one status per servo.
  std::vector<ServoStatus> update(const std::vector<ServoReading> & readings, double t);

  // Call periodically: latches a fault if telemetry is stale (or never arrived).
  void check_timeout(double t);

  // Overall level: FAULT while latched, otherwise the level of the latest sample.
  Level level() const;
  bool faulted() const {return latched_fault_;}
  const std::string & fault_reason() const {return fault_reason_;}

  // Clear a latched fault. Only allowed if fresh telemetry shows no fault right now.
  bool reset(double t, std::string & why_not);

private:
  void latch(const std::string & reason);

  Limits lim_;
  double start_t_ = -1.0;
  double last_rx_t_ = -1.0;
  Level current_ = Level::OK;
  bool latched_fault_ = false;
  std::string fault_reason_;
  std::map<std::string, double> overload_since_;  // servo -> time the overload started
};

}  // namespace so101_safety
