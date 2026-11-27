// feetech_cli - talk to a Feetech STS bus (real /dev/ttyACM0 or the virtual /tmp/so101_bus).
//
//   feetech_cli [--device PATH] scan [max_id]          ping IDs 1..max_id, print model numbers
//   feetech_cli [--device PATH] read ID REGISTER        e.g. read 2 Present_Position
//   feetech_cli [--device PATH] write ID REGISTER VALUE e.g. write 1 Goal_Position 2300
//   feetech_cli [--device PATH] loop [seconds]          safe start + control loop benchmark
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "feetech_bus/bus.hpp"

using feetech::FeetechBus;
using feetech::Status;
namespace reg = feetech::reg;
using Clock = std::chrono::steady_clock;

namespace
{
const std::vector<uint8_t> kIds{1, 2, 3, 4, 5, 6};
// ready pose of the SO-101 in encoder steps (= so101_sim.scene.ready_q(), gripper 0)
const std::vector<int32_t> kReady{2046, 1919, 2576, 2673, 2078, 2048};

int usage()
{
  std::fprintf(stderr,
    "usage: feetech_cli [--device PATH] scan [max_id] | read ID REG | write ID REG VALUE | loop [s]\n");
  return 2;
}

double seconds_since(Clock::time_point t0)
{
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::vector<std::pair<uint8_t, int32_t>> pairs(const std::vector<int32_t> & v)
{
  std::vector<std::pair<uint8_t, int32_t>> out;
  for (std::size_t k = 0; k < kIds.size(); ++k) {out.emplace_back(kIds[k], v[k]);}
  return out;
}

int loop(FeetechBus & bus, double seconds)
{
  // safe start: hold the CURRENT position before switching the torque on (the servo keeps its
  // old goal in RAM and would jump to it), then ramp smoothly to the ready pose
  std::vector<int32_t> here;
  if (bus.sync_read(reg::kPresentPosition, kIds, &here) != Status::kOk) {
    std::fprintf(stderr, "cannot read the present positions\n");
    return 1;
  }
  bus.sync_write(reg::kGoalPosition, pairs(here));
  std::vector<std::pair<uint8_t, int32_t>> on;
  for (auto id : kIds) {on.emplace_back(id, 1);}
  bus.sync_write(reg::kTorqueEnable, on);
  const auto t_ramp = Clock::now();
  for (double s = 0; (s = seconds_since(t_ramp) / 2.0) < 1.0; ) {
    const double w = 0.5 - 0.5 * std::cos(M_PI * s);
    std::vector<int32_t> g(kIds.size());
    for (std::size_t k = 0; k < kIds.size(); ++k) {
      g[k] = static_cast<int32_t>(std::lround(here[k] + w * (kReady[k] - here[k])));
    }
    bus.sync_write(reg::kGoalPosition, pairs(g));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  bus.sync_write(reg::kGoalPosition, pairs(kReady));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // control loop: SYNC_WRITE goals + SYNC_READ positions, as fast as the bus allows
  std::vector<double> latency_ms;
  std::vector<int32_t> pos;
  double err2 = 0;
  long n_err = 0;
  const auto t0 = Clock::now();
  double t;
  while ((t = seconds_since(t0)) < seconds) {
    const int wave = static_cast<int>(300 * std::sin(2 * M_PI * t / 3.0));
    std::vector<int32_t> goal = kReady;
    goal[1] += wave;
    goal[2] -= wave;
    bus.sync_write(reg::kGoalPosition, pairs(goal));
    const auto t1 = Clock::now();
    if (bus.sync_read(reg::kPresentPosition, kIds, &pos) == Status::kOk) {
      latency_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t1).count());
      err2 += std::pow(goal[1] - pos[1], 2);
      ++n_err;
    }
  }
  bus.sync_write(reg::kGoalPosition, pairs(kReady));
  std::sort(latency_ms.begin(), latency_ms.end());
  const auto pct = [&](double q) {return latency_ms[static_cast<std::size_t>(q * (latency_ms.size() - 1))];};
  const auto & st = bus.stats();
  std::printf("control loop: %.0f Hz, SYNC_READ median %.2f ms, p99 %.2f ms, max %.2f ms, timeouts %llu\n",
    latency_ms.size() / seconds, pct(0.5), pct(0.99), latency_ms.back(),
    static_cast<unsigned long long>(st.timeouts));
  std::printf("shoulder_lift tracking error: rms %.1f steps\n", std::sqrt(err2 / std::max(1L, n_err)));
  return 0;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string device = "/tmp/so101_bus";
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() >= 2 && args[0] == "--device") {
    device = args[1];
    args.erase(args.begin(), args.begin() + 2);
  }
  if (args.empty()) {return usage();}

  FeetechBus bus(std::chrono::milliseconds(20));
  std::string error;
  if (!bus.open(device, 1000000, &error)) {
    std::fprintf(stderr, "cannot open %s: %s\n", device.c_str(), error.c_str());
    return 1;
  }

  const std::string cmd = args[0];
  if (cmd == "scan") {
    const int max_id = args.size() > 1 ? std::atoi(args[1].c_str()) : 10;
    for (int id = 1; id <= max_id; ++id) {
      int32_t model = 0;
      const Status s = bus.read(static_cast<uint8_t>(id), reg::kModelNumber, &model);
      if (s == Status::kOk) {
        std::printf("ID %2d: model %d%s\n", id, model, model == feetech::kSts3215ModelNumber ? " (STS3215)" : "");
      }
    }
    return 0;
  }
  if ((cmd == "read" && args.size() == 3) || (cmd == "write" && args.size() == 4)) {
    const auto id = static_cast<uint8_t>(std::atoi(args[1].c_str()));
    const feetech::Register * r = feetech::find_register(args[2]);
    if (!r) {
      std::fprintf(stderr, "unknown register %s\n", args[2].c_str());
      return 2;
    }
    uint8_t err_byte = 0;
    Status s;
    int32_t value = 0;
    if (cmd == "read") {
      s = bus.read(id, *r, &value, &err_byte);
    } else {
      value = std::atoi(args[3].c_str());
      s = bus.write(id, *r, value, &err_byte);
    }
    if (s != Status::kOk) {
      std::fprintf(stderr, "%s failed: %s\n", cmd.c_str(), feetech::to_string(s));
      return 1;
    }
    if (cmd == "write" && (err_byte & feetech::err::kInstruction)) {
      std::printf("ID %d refused the write to %s (error byte 0x%02X)%s\n", id, r->name, err_byte,
        r->address < 40 ? " - EEPROM is locked: write Lock 0 first" : "");
      return 1;
    }
    std::printf("ID %d %s = %d   (error byte 0x%02X)\n", id, r->name, value, err_byte);
    return 0;
  }
  if (cmd == "loop") {
    return loop(bus, args.size() > 1 ? std::atof(args[1].c_str()) : 6.0);
  }
  return usage();
}
