// Transactions on a half-duplex Feetech bus: one request out, the expected reply (or replies)
// back within a timeout - or a clear error. Never throws in the control loop.
//
// Rules a driver must follow on this bus:
//   * only one transaction at a time (the wire is shared in both directions)
//   * broadcast (ID 0xFE) and SYNC_WRITE get no reply - do not wait for one
//   * a missing reply is normal (unplugged servo, noise): time out, report, let the caller decide
//   * a reply can arrive LATE, after its request already timed out. It must not be taken as the
//     answer to the next request: stale input is thrown away before every request, and every
//     reply is checked for the expected ID and length.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "feetech_bus/protocol.hpp"
#include "feetech_bus/serial_port.hpp"

namespace feetech
{

enum class Status { kOk, kTimeout, kIoError, kBadReply };
const char * to_string(Status s);

struct BusStats
{
  uint64_t transactions = 0;
  uint64_t timeouts = 0;
  uint64_t bad_replies = 0;
  uint64_t io_errors = 0;
};

class FeetechBus
{
public:
  explicit FeetechBus(std::chrono::microseconds timeout = std::chrono::milliseconds(20))
  : timeout_(timeout) {}

  bool open(const std::string & device, int baud = 1000000, std::string * error = nullptr);
  void close() {port_.close();}
  bool is_open() const {return port_.is_open();}
  void set_timeout(std::chrono::microseconds t) {timeout_ = t;}

  Status ping(uint8_t id, uint8_t * error_byte = nullptr);
  Status read(uint8_t id, const Register & r, int32_t * value, uint8_t * error_byte = nullptr);
  Status write(uint8_t id, const Register & r, int32_t value, uint8_t * error_byte = nullptr);
  // one packet for all servos, no reply
  Status sync_write(const Register & r, const std::vector<std::pair<uint8_t, int32_t>> & values);
  // values[k] belongs to ids[k]; error_bits = OR of all servos' error bytes
  Status sync_read(
    const Register & r, const std::vector<uint8_t> & ids, std::vector<int32_t> * values,
    uint8_t * error_bits = nullptr);
  // raw bytes [address, address + length) of every servo; data[k] belongs to ids[k].
  // Decode a register inside the block with decode_value(reg, data[k].data() + reg.address - address).
  Status sync_read_block(
    uint8_t address, uint8_t length, const std::vector<uint8_t> & ids,
    std::vector<Bytes> * data, uint8_t * error_bits = nullptr);

  const BusStats & stats() const {return stats_;}

private:
  Status transact(const Bytes & request, std::size_t n_replies, std::vector<Packet> * replies);

  SerialPort port_;
  PacketParser parser_;
  std::chrono::microseconds timeout_;
  BusStats stats_;
  uint8_t rx_[512];
};

}  // namespace feetech
