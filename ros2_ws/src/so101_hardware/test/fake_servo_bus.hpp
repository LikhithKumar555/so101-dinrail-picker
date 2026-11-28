// Fake STS3215 servos behind a real pseudo-terminal, for testing the driver without Python or
// MuJoCo. Each servo has a 256-byte register memory and answers PING, READ, WRITE, SYNC_WRITE and
// SYNC_READ like the real one. Deliberately simple "physics": with torque on, a servo is at its
// goal at once - enough to see whether a driver would make the arm jump.
#pragma once

#include <poll.h>
#include <pty.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "feetech_bus/protocol.hpp"

namespace so101_hardware_test
{

namespace reg = feetech::reg;

class FakeServoBus
{
public:
  explicit FakeServoBus(const std::vector<uint8_t> & ids)
  {
    termios tio{};
    cfmakeraw(&tio);
    if (openpty(&master_, &slave_, name_, &tio, nullptr) != 0) {
      throw std::runtime_error("openpty failed");
    }
    for (uint8_t id : ids) {
      auto & m = mem_[id];
      m.fill(0);
      put(m, reg::kModelNumber, feetech::kSts3215ModelNumber);
      put(m, reg::kId, id);
      put(m, reg::kPresentPosition, 2048);
      put(m, reg::kGoalPosition, 2048);
      put(m, reg::kPresentVoltage, 74);        // 7.4 V
      put(m, reg::kPresentTemperature, 31);
    }
    thread_ = std::thread([this] {run();});
  }

  ~FakeServoBus()
  {
    stop_ = true;
    thread_.join();
    ::close(master_);
    ::close(slave_);
  }

  std::string device() const {return name_;}

  // --- test controls (thread-safe) ---
  void set(uint8_t id, const feetech::Register & r, int32_t value)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    put(mem_.at(id), r, value);
  }
  int32_t get(uint8_t id, const feetech::Register & r)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    return feetech::decode_value(r, mem_.at(id).data() + r.address);
  }
  void set_error_byte(uint8_t id, uint8_t bits)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    error_[id] = bits;
  }
  void unplug(uint8_t id)                     // the servo stays silent from now on
  {
    std::lock_guard<std::mutex> lk(mutex_);
    unplugged_.insert(id);
  }
  void ignore_sync_writes(bool ignore) {ignore_sync_writes_ = ignore;}   // a lost broadcast
  void drop_replies(int n) {drop_replies_ = n;}   // the next n answered requests get no reply
  int sync_writes() const {return sync_writes_;}

private:
  static void put(std::array<uint8_t, 256> & m, const feetech::Register & r, int32_t value)
  {
    const auto b = feetech::encode_value(r, value);
    std::copy(b.begin(), b.end(), m.begin() + r.address);
  }

  bool answers(uint8_t id) const {return mem_.count(id) && !unplugged_.count(id);}

  // writes go through here: torque on means "the servo is at its goal now"
  void write_mem(uint8_t id, uint8_t addr, const uint8_t * data, std::size_t n)
  {
    auto & m = mem_.at(id);
    std::copy(data, data + n, m.begin() + addr);
    if (m[reg::kTorqueEnable.address]) {
      m[reg::kPresentPosition.address] = m[reg::kGoalPosition.address];
      m[reg::kPresentPosition.address + 1] = m[reg::kGoalPosition.address + 1];
    }
  }

  void run()
  {
    feetech::PacketParser parser;
    std::array<uint8_t, 512> buf{};
    while (!stop_) {
      pollfd p{master_, POLLIN, 0};
      if (poll(&p, 1, 5) <= 0) {continue;}
      const ssize_t n = ::read(master_, buf.data(), buf.size());
      if (n <= 0) {continue;}
      for (const auto & pkt : parser.feed(buf.data(), static_cast<std::size_t>(n))) {
        const feetech::Bytes out = handle(pkt);
        if (out.empty()) {continue;}
        if (drop_replies_ > 0) {     // reply lost on the wire (or far too late)
          --drop_replies_;
          continue;
        }
        (void)!::write(master_, out.data(), out.size());
      }
    }
  }

  feetech::Bytes handle(const feetech::Packet & pkt)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    feetech::Bytes out;
    auto reply = [&](uint8_t id, const feetech::Bytes & params) {
        const auto r = feetech::make_packet(id, error_[id], params);
        out.insert(out.end(), r.begin(), r.end());
      };
    const auto & prm = pkt.params;
    if (pkt.code == feetech::kSyncRead) {
      const uint8_t addr = prm[0], size = prm[1];
      for (std::size_t k = 2; k < prm.size(); ++k) {
        if (answers(prm[k])) {
          const auto & m = mem_.at(prm[k]);
          reply(prm[k], feetech::Bytes(m.begin() + addr, m.begin() + addr + size));
        }
      }
    } else if (pkt.code == feetech::kSyncWrite) {
      ++sync_writes_;
      if (ignore_sync_writes_) {return out;}
      const uint8_t addr = prm[0], size = prm[1];
      for (std::size_t k = 2; k + size < prm.size(); k += 1 + size) {   // entry: ID + size bytes
        if (mem_.count(prm[k])) {write_mem(prm[k], addr, prm.data() + k + 1, size);}
      }
    } else if (answers(pkt.id)) {
      auto & m = mem_.at(pkt.id);
      if (pkt.code == feetech::kPing) {
        reply(pkt.id, {});
      } else if (pkt.code == feetech::kRead) {
        reply(pkt.id, feetech::Bytes(m.begin() + prm[0], m.begin() + prm[0] + prm[1]));
      } else if (pkt.code == feetech::kWrite) {
        write_mem(pkt.id, prm[0], prm.data() + 1, prm.size() - 1);
        reply(pkt.id, {});
      }
    }
    return out;
  }

  int master_ = -1, slave_ = -1;
  char name_[128]{};
  std::mutex mutex_;
  std::map<uint8_t, std::array<uint8_t, 256>> mem_;
  std::map<uint8_t, uint8_t> error_;
  std::set<uint8_t> unplugged_;
  std::atomic<bool> ignore_sync_writes_{false};
  std::atomic<int> sync_writes_{0};
  std::atomic<int> drop_replies_{0};
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

}  // namespace so101_hardware_test
