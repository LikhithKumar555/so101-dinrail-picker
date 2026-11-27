#include "feetech_bus/bus.hpp"

namespace feetech
{

const char * to_string(Status s)
{
  switch (s) {
    case Status::kOk: return "ok";
    case Status::kTimeout: return "timeout";
    case Status::kIoError: return "I/O error";
    case Status::kBadReply: return "bad reply";
  }
  return "?";
}

bool FeetechBus::open(const std::string & device, int baud, std::string * error)
{
  parser_.reset();
  return port_.open(device, baud, error);
}

Status FeetechBus::transact(const Bytes & request, std::size_t n_replies, std::vector<Packet> * replies)
{
  ++stats_.transactions;
  // a late reply to an earlier, timed-out request must not be read as the answer to this one
  port_.discard_input();
  parser_.reset();
  if (!port_.write_all(request.data(), request.size())) {
    ++stats_.io_errors;
    return Status::kIoError;
  }
  replies->clear();
  if (n_replies == 0) {
    return Status::kOk;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout_;
  while (replies->size() < n_replies) {
    const auto left = std::chrono::duration_cast<std::chrono::microseconds>(
      deadline - std::chrono::steady_clock::now());
    if (left.count() <= 0) {
      ++stats_.timeouts;
      return Status::kTimeout;
    }
    const ssize_t n = port_.read_some(rx_, sizeof(rx_), left);
    if (n < 0) {
      ++stats_.io_errors;
      return Status::kIoError;
    }
    if (n > 0) {
      for (auto & p : parser_.feed(rx_, static_cast<std::size_t>(n))) {
        replies->push_back(std::move(p));
      }
    }
  }
  return Status::kOk;
}

Status FeetechBus::ping(uint8_t id, uint8_t * error_byte)
{
  std::vector<Packet> r;
  const Status s = transact(feetech::ping(id), 1, &r);
  if (s != Status::kOk) {return s;}
  if (r[0].id != id) {
    ++stats_.bad_replies;
    return Status::kBadReply;
  }
  if (error_byte) {*error_byte = r[0].code;}
  return Status::kOk;
}

Status FeetechBus::read(uint8_t id, const Register & reg, int32_t * value, uint8_t * error_byte)
{
  std::vector<Packet> r;
  const Status s = transact(feetech::read(id, reg), 1, &r);
  if (s != Status::kOk) {return s;}
  if (r[0].id != id || r[0].params.size() != reg.size) {
    ++stats_.bad_replies;
    return Status::kBadReply;
  }
  *value = decode_value(reg, r[0].params.data());
  if (error_byte) {*error_byte = r[0].code;}
  return Status::kOk;
}

Status FeetechBus::write(uint8_t id, const Register & reg, int32_t value, uint8_t * error_byte)
{
  std::vector<Packet> r;
  const bool broadcast = id == kBroadcastId;
  const Status s = transact(feetech::write(id, reg, value), broadcast ? 0 : 1, &r);
  if (s != Status::kOk || broadcast) {return s;}
  if (error_byte) {*error_byte = r[0].code;}
  return Status::kOk;
}

Status FeetechBus::sync_write(const Register & reg, const std::vector<std::pair<uint8_t, int32_t>> & values)
{
  std::vector<Packet> r;
  return transact(feetech::sync_write(reg, values), 0, &r);
}

Status FeetechBus::sync_read(
  const Register & reg, const std::vector<uint8_t> & ids, std::vector<int32_t> * values,
  uint8_t * error_bits)
{
  std::vector<Packet> r;
  const Status s = transact(feetech::sync_read(reg, ids), ids.size(), &r);
  if (s != Status::kOk) {return s;}
  values->assign(ids.size(), 0);
  uint8_t bits = 0;
  for (std::size_t k = 0; k < ids.size(); ++k) {
    // each servo answers in the listed order; check anyway
    const Packet & p = r[k];
    if (p.id != ids[k] || p.params.size() != reg.size) {
      ++stats_.bad_replies;
      return Status::kBadReply;
    }
    (*values)[k] = decode_value(reg, p.params.data());
    bits |= p.code;
  }
  if (error_bits) {*error_bits = bits;}
  return Status::kOk;
}

}  // namespace feetech
