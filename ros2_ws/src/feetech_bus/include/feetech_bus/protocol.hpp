// Feetech STS serial bus protocol (STS3215) - packets, checksums, control table, stream parser.
//
//   instruction (host -> servo):  0xFF 0xFF  ID  LEN  INSTR  PARAM_1..PARAM_N  CHECKSUM
//   status      (servo -> host):  0xFF 0xFF  ID  LEN  ERROR  PARAM_1..PARAM_N  CHECKSUM
//   LEN = N + 2,   CHECKSUM = ~(ID + LEN + INSTR/ERROR + sum(PARAM)) & 0xFF
//
// Multi-byte registers are little-endian. Signed registers use SIGN-MAGNITUDE (one sign bit,
// the rest is the magnitude), not two's complement. Same definitions and test vectors as the
// Python reference (sim/so101_sim/feetech/protocol.py).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace feetech
{

using Bytes = std::vector<uint8_t>;

constexpr uint8_t kBroadcastId = 0xFE;
constexpr int kStepsPerRev = 4096;
constexpr int kSts3215ModelNumber = 777;

enum Instruction : uint8_t
{
  kPing = 0x01, kRead = 0x02, kWrite = 0x03, kRegWrite = 0x04, kAction = 0x05,
  kSyncRead = 0x82, kSyncWrite = 0x83,
};

// bits of the status (error) byte in every reply
namespace err
{
constexpr uint8_t kVoltage = 1 << 0, kAngle = 1 << 1, kOverheat = 1 << 2, kOverCurrent = 1 << 3,
  kChecksum = 1 << 4, kOverload = 1 << 5, kInstruction = 1 << 6;
}

struct Register
{
  const char * name;
  uint8_t address;
  uint8_t size;      // bytes: 1 or 2
  int8_t sign_bit;   // < 0: unsigned
};

namespace reg
{
constexpr Register kModelNumber{"Model_Number", 3, 2, -1};
constexpr Register kId{"ID", 5, 1, -1};
constexpr Register kReturnDelayTime{"Return_Delay_Time", 7, 1, -1};
constexpr Register kMinPositionLimit{"Min_Position_Limit", 9, 2, -1};
constexpr Register kMaxPositionLimit{"Max_Position_Limit", 11, 2, -1};
constexpr Register kMaxTemperatureLimit{"Max_Temperature_Limit", 13, 1, -1};
constexpr Register kPCoefficient{"P_Coefficient", 21, 1, -1};
constexpr Register kDCoefficient{"D_Coefficient", 22, 1, -1};
constexpr Register kICoefficient{"I_Coefficient", 23, 1, -1};
constexpr Register kHomingOffset{"Homing_Offset", 31, 2, 11};
constexpr Register kTorqueEnable{"Torque_Enable", 40, 1, -1};
constexpr Register kAcceleration{"Acceleration", 41, 1, -1};
constexpr Register kGoalPosition{"Goal_Position", 42, 2, -1};
constexpr Register kGoalVelocity{"Goal_Velocity", 46, 2, 15};
constexpr Register kTorqueLimit{"Torque_Limit", 48, 2, -1};
constexpr Register kLock{"Lock", 55, 1, -1};
constexpr Register kPresentPosition{"Present_Position", 56, 2, -1};
constexpr Register kPresentVelocity{"Present_Velocity", 58, 2, 15};
constexpr Register kPresentLoad{"Present_Load", 60, 2, 10};
constexpr Register kPresentVoltage{"Present_Voltage", 62, 1, -1};
constexpr Register kPresentTemperature{"Present_Temperature", 63, 1, -1};
constexpr Register kStatus{"Status", 65, 1, -1};
constexpr Register kMoving{"Moving", 66, 1, -1};
constexpr Register kPresentCurrent{"Present_Current", 69, 2, -1};
}  // namespace reg

// register by its name (as in the Feetech/LeRobot tables), nullptr if unknown
const Register * find_register(const std::string & name);

uint8_t checksum(const uint8_t * body, std::size_t n);
Bytes make_packet(uint8_t id, uint8_t code, const Bytes & params = {});

uint32_t encode_sign_magnitude(int32_t value, int sign_bit);   // throws std::out_of_range
int32_t decode_sign_magnitude(uint32_t raw, int sign_bit);
Bytes encode_value(const Register & r, int32_t value);         // little-endian
int32_t decode_value(const Register & r, const uint8_t * data);

// instruction builders
Bytes ping(uint8_t id);
Bytes read(uint8_t id, const Register & r);
Bytes write(uint8_t id, const Register & r, int32_t value);
Bytes sync_write(const Register & r, const std::vector<std::pair<uint8_t, int32_t>> & values);
Bytes sync_read(const Register & r, const std::vector<uint8_t> & ids);

struct Packet
{
  uint8_t id = 0;
  uint8_t code = 0;   // instruction (to a servo) or error byte (from a servo)
  Bytes params;
};

// Byte-stream parser: a serial read can return half a packet, two packets, or garbage.
// feed() returns every complete packet with a valid checksum; broken ones are counted and skipped.
class PacketParser
{
public:
  std::vector<Packet> feed(const uint8_t * data, std::size_t n);
  void reset() {buf_.clear();}
  std::size_t bad_checksums() const {return bad_checksums_;}

private:
  Bytes buf_;
  std::size_t bad_checksums_ = 0;
};

// time on the wire at 8N1: 10 bits per byte
inline double transfer_time_s(std::size_t n_bytes, int baud) {return n_bytes * 10.0 / baud;}

}  // namespace feetech
