#include "so101_safety/safety_logic.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace so101_safety
{

const char * to_string(Level level)
{
  switch (level) {
    case Level::OK: return "OK";
    case Level::WARN: return "WARN";
    case Level::FAULT: return "FAULT";
  }
  return "?";
}

SafetyLogic::SafetyLogic(const Limits & limits)
: lim_(limits) {}

std::vector<ServoStatus> SafetyLogic::update(const std::vector<ServoReading> & readings, double t)
{
  last_rx_t_ = t;
  current_ = Level::OK;
  std::vector<ServoStatus> out;
  out.reserve(readings.size());

  for (const auto & r : readings) {
    ServoStatus s{r.name, Level::OK, "ok"};
    std::ostringstream msg;
    auto raise = [&](Level l, const std::string & text) {
        if (l > s.level) {s.level = l; s.message = text;}
      };

    // Temperature: the servo's motor windings and electronics overheat and get damaged.
    if (r.temperature_c >= lim_.temp_fault_c) {
      msg << "temperature " << r.temperature_c << " C >= " << lim_.temp_fault_c << " C";
      raise(Level::FAULT, msg.str());
    } else if (r.temperature_c >= lim_.temp_warn_c) {
      msg << "temperature " << r.temperature_c << " C >= " << lim_.temp_warn_c << " C";
      raise(Level::WARN, msg.str());
    }
    msg.str("");

    // Load: a sustained high load means the arm is blocked / colliding / lifting too much.
    const double load = std::abs(r.load);
    if (load >= lim_.load_fault) {
      auto it = overload_since_.find(r.name);
      if (it == overload_since_.end()) {
        it = overload_since_.emplace(r.name, t).first;
      }
      const double held = t - it->second;
      msg << "load " << load << " >= " << lim_.load_fault << " for " << held << " s";
      raise(held >= lim_.load_hold_s ? Level::FAULT : Level::WARN, msg.str());
    } else {
      overload_since_.erase(r.name);
      if (load >= lim_.load_warn) {
        msg << "load " << load << " >= " << lim_.load_warn;
        raise(Level::WARN, msg.str());
      }
    }
    msg.str("");

    // Supply voltage.
    if (r.voltage_v < lim_.voltage_min_v || r.voltage_v > lim_.voltage_max_v) {
      msg << "voltage " << r.voltage_v << " V outside [" << lim_.voltage_min_v << ", " <<
        lim_.voltage_max_v << "] V";
      raise(Level::FAULT, msg.str());
    }

    if (s.level == Level::FAULT && !latched_fault_) {
      latch(r.name + ": " + s.message);
    }
    current_ = std::max(current_, s.level);
    out.push_back(s);
  }
  return out;
}

void SafetyLogic::check_timeout(double t)
{
  if (start_t_ < 0.0) {
    start_t_ = t;
  }
  const bool never = last_rx_t_ < 0.0;
  const double ref = never ? start_t_ : last_rx_t_;
  const double limit = never ? lim_.startup_timeout_s : lim_.timeout_s;
  if (t - ref > limit && !latched_fault_) {
    std::ostringstream msg;
    msg << (last_rx_t_ < 0.0 ? "no telemetry received" : "telemetry timeout") << " (" <<
      (t - ref) << " s > " << limit << " s)";
    latch(msg.str());
  }
}

Level SafetyLogic::level() const
{
  return latched_fault_ ? Level::FAULT : current_;
}

bool SafetyLogic::reset(double t, std::string & why_not)
{
  if (last_rx_t_ < 0.0 || t - last_rx_t_ > lim_.timeout_s) {
    why_not = "telemetry is not fresh";
    return false;
  }
  if (current_ == Level::FAULT) {
    why_not = "fault condition is still present";
    return false;
  }
  latched_fault_ = false;
  fault_reason_.clear();
  why_not.clear();
  return true;
}

void SafetyLogic::latch(const std::string & reason)
{
  latched_fault_ = true;
  fault_reason_ = reason;
}

}  // namespace so101_safety
