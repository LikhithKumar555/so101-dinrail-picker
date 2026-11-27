#include "feetech_bus/protocol.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>

namespace feetech
{

const Register * find_register(const std::string & name)
{
  static const std::array<const Register *, 24> all = {
    &reg::kModelNumber, &reg::kId, &reg::kReturnDelayTime, &reg::kMinPositionLimit,
    &reg::kMaxPositionLimit, &reg::kMaxTemperatureLimit, &reg::kPCoefficient, &reg::kDCoefficient,
    &reg::kICoefficient, &reg::kHomingOffset, &reg::kTorqueEnable, &reg::kAcceleration,
    &reg::kGoalPosition, &reg::kGoalVelocity, &reg::kTorqueLimit, &reg::kLock,
    &reg::kPresentPosition, &reg::kPresentVelocity, &reg::kPresentLoad, &reg::kPresentVoltage,
    &reg::kPresentTemperature, &reg::kStatus, &reg::kMoving, &reg::kPresentCurrent};
  for (const Register * r : all) {
    if (name == r->name) {
      return r;
    }
  }
  return nullptr;
}

uint8_t checksum(const uint8_t * body, std::size_t n)
{
  unsigned sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += body[i];
  }
  return static_cast<uint8_t>(~sum & 0xFF);
}

Bytes make_packet(uint8_t id, uint8_t code, const Bytes & params)
{
  Bytes p;
  p.reserve(params.size() + 6);
  p.push_back(0xFF);
  p.push_back(0xFF);
  p.push_back(id);
  p.push_back(static_cast<uint8_t>(params.size() + 2));
  p.push_back(code);
  p.insert(p.end(), params.begin(), params.end());
  p.push_back(checksum(p.data() + 2, p.size() - 2));
  return p;
}

uint32_t encode_sign_magnitude(int32_t value, int sign_bit)
{
  const uint32_t magnitude = static_cast<uint32_t>(std::abs(value));
  if (magnitude >= (1u << sign_bit)) {
    throw std::out_of_range("value does not fit below the sign bit");
  }
  return magnitude | (value < 0 ? (1u << sign_bit) : 0u);
}

int32_t decode_sign_magnitude(uint32_t raw, int sign_bit)
{
  const int32_t magnitude = static_cast<int32_t>(raw & ((1u << sign_bit) - 1));
  return (raw & (1u << sign_bit)) ? -magnitude : magnitude;
}

Bytes encode_value(const Register & r, int32_t value)
{
  const uint32_t raw = r.sign_bit >= 0 ? encode_sign_magnitude(value, r.sign_bit) :
    static_cast<uint32_t>(value);
  Bytes out(r.size);
  for (int i = 0; i < r.size; ++i) {
    out[i] = static_cast<uint8_t>((raw >> (8 * i)) & 0xFF);   // low byte first
  }
  return out;
}

int32_t decode_value(const Register & r, const uint8_t * data)
{
  uint32_t raw = 0;
  for (int i = 0; i < r.size; ++i) {
    raw |= static_cast<uint32_t>(data[i]) << (8 * i);
  }
  return r.sign_bit >= 0 ? decode_sign_magnitude(raw, r.sign_bit) : static_cast<int32_t>(raw);
}

Bytes ping(uint8_t id) {return make_packet(id, kPing);}

Bytes read(uint8_t id, const Register & r) {return make_packet(id, kRead, {r.address, r.size});}

Bytes write(uint8_t id, const Register & r, int32_t value)
{
  Bytes params{r.address};
  const Bytes v = encode_value(r, value);
  params.insert(params.end(), v.begin(), v.end());
  return make_packet(id, kWrite, params);
}

Bytes sync_write(const Register & r, const std::vector<std::pair<uint8_t, int32_t>> & values)
{
  Bytes params{r.address, r.size};
  for (const auto & [id, value] : values) {
    params.push_back(id);
    const Bytes v = encode_value(r, value);
    params.insert(params.end(), v.begin(), v.end());
  }
  return make_packet(kBroadcastId, kSyncWrite, params);
}

Bytes sync_read(const Register & r, const std::vector<uint8_t> & ids)
{
  Bytes params{r.address, r.size};
  params.insert(params.end(), ids.begin(), ids.end());
  return make_packet(kBroadcastId, kSyncRead, params);
}

std::vector<Packet> PacketParser::feed(const uint8_t * data, std::size_t n)
{
  buf_.insert(buf_.end(), data, data + n);
  std::vector<Packet> out;
  std::size_t pos = 0;   // consume from the front; erase once at the end
  while (true) {
    // find the 0xFF 0xFF header
    std::size_t start = pos;
    while (start + 1 < buf_.size() && !(buf_[start] == 0xFF && buf_[start + 1] == 0xFF)) {
      ++start;
    }
    if (start + 1 >= buf_.size()) {
      // keep a trailing 0xFF: it may be the first half of the next header
      pos = (!buf_.empty() && buf_.back() == 0xFF) ? buf_.size() - 1 : buf_.size();
      break;
    }
    pos = start;
    if (buf_.size() - pos < 4) {
      break;
    }
    if (buf_[pos + 2] == 0xFF) {   // FF FF FF: the header starts one byte later
      ++pos;
      continue;
    }
    const std::size_t length = buf_[pos + 3];
    if (length < 2) {
      pos += 2;
      continue;
    }
    const std::size_t total = 4 + length;
    if (buf_.size() - pos < total) {
      break;   // wait for the rest of this packet
    }
    const uint8_t * body = buf_.data() + pos + 2;   // ID, LEN, CODE, PARAMS
    if (checksum(body, total - 3) == buf_[pos + total - 1]) {
      Packet p;
      p.id = body[0];
      p.code = body[2];
      p.params.assign(body + 3, body + total - 3);
      out.push_back(std::move(p));
      pos += total;
    } else {
      ++bad_checksums_;
      pos += 2;   // skip this header and look for the next one
    }
  }
  buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(pos));
  return out;
}

}  // namespace feetech
