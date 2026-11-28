// FeetechBus over a real pseudo-terminal, against a small fake servo bus running in a thread.
#include <gtest/gtest.h>

#include <poll.h>
#include <pty.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <stdexcept>
#include <thread>

#include "feetech_bus/bus.hpp"

using namespace std::chrono_literals;
namespace reg = feetech::reg;

namespace
{
// Fake servos with IDs 1..3: register memory, PING/READ/WRITE/SYNC_READ, silence otherwise.
class FakeServos
{
public:
  FakeServos()
  {
    termios tio{};
    cfmakeraw(&tio);
    if (openpty(&master_, &slave_, name_, &tio, nullptr) != 0) {throw std::runtime_error("openpty");}
    for (uint8_t id = 1; id <= 3; ++id) {
      auto & m = mem_[id];
      m.fill(0);
      m[3] = 777 & 0xFF;                          // Model_Number, little-endian
      m[4] = 777 >> 8;
      m[5] = id;
      m[56] = static_cast<uint8_t>(id);          // Present_Position = 0x0800 + id = 2048 + id
      m[57] = 0x08;
    }
    thread_ = std::thread([this] {run();});
  }
  ~FakeServos()
  {
    stop_ = true;
    thread_.join();
    ::close(master_);
    ::close(slave_);
  }
  std::string device() const {return name_;}
  std::atomic<int> delay_next_reply_ms{0};      // to provoke a late reply

private:
  void run()
  {
    feetech::PacketParser parser;
    std::array<uint8_t, 256> buf{};
    while (!stop_) {
      pollfd p{master_, POLLIN, 0};
      if (poll(&p, 1, 5) <= 0) {continue;}
      const ssize_t n = ::read(master_, buf.data(), buf.size());
      if (n <= 0) {continue;}
      for (const auto & pkt : parser.feed(buf.data(), static_cast<std::size_t>(n))) {
        feetech::Bytes out;
        auto reply = [&](uint8_t id, const feetech::Bytes & params) {
            const auto r = feetech::make_packet(id, 0, params);
            out.insert(out.end(), r.begin(), r.end());
          };
        if (pkt.code == feetech::kSyncRead) {
          const uint8_t addr = pkt.params[0], size = pkt.params[1];
          for (std::size_t k = 2; k < pkt.params.size(); ++k) {
            const uint8_t id = pkt.params[k];
            if (mem_.count(id)) {
              reply(id, feetech::Bytes(mem_[id].begin() + addr, mem_[id].begin() + addr + size));
            }
          }
        } else if (mem_.count(pkt.id)) {
          auto & m = mem_[pkt.id];
          if (pkt.code == feetech::kPing) {
            reply(pkt.id, {});
          } else if (pkt.code == feetech::kRead) {
            reply(pkt.id, feetech::Bytes(m.begin() + pkt.params[0], m.begin() + pkt.params[0] + pkt.params[1]));
          } else if (pkt.code == feetech::kWrite) {
            std::copy(pkt.params.begin() + 1, pkt.params.end(), m.begin() + pkt.params[0]);
            reply(pkt.id, {});
          }
        }
        if (out.empty()) {continue;}
        const int delay = delay_next_reply_ms.exchange(0);
        if (delay > 0) {std::this_thread::sleep_for(std::chrono::milliseconds(delay));}
        (void)!::write(master_, out.data(), out.size());
      }
    }
  }

  int master_ = -1, slave_ = -1;
  char name_[128]{};
  std::map<uint8_t, std::array<uint8_t, 256>> mem_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

class BusPty : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::string error;
    ASSERT_TRUE(bus.open(fake.device(), 1000000, &error)) << error;
  }
  FakeServos fake;
  feetech::FeetechBus bus{20ms};
};
}  // namespace

TEST_F(BusPty, PingAndModelNumber)
{
  uint8_t e = 0xFF;
  EXPECT_EQ(bus.ping(2, &e), feetech::Status::kOk);
  EXPECT_EQ(e, 0);
  int32_t model = 0;
  ASSERT_EQ(bus.read(1, reg::kModelNumber, &model), feetech::Status::kOk);
  EXPECT_EQ(model, feetech::kSts3215ModelNumber);
}

TEST_F(BusPty, UnknownIdTimesOutAndTheBusRecovers)
{
  EXPECT_EQ(bus.ping(42), feetech::Status::kTimeout);
  EXPECT_EQ(bus.stats().timeouts, 1u);
  EXPECT_EQ(bus.ping(1), feetech::Status::kOk);
}

TEST_F(BusPty, RawModeKeepsControlCharacters)
{
  // Goal_Position 3331 = 0x0D03: carriage return and Ctrl-C on the wire
  ASSERT_EQ(bus.write(1, reg::kGoalPosition, 3331), feetech::Status::kOk);
  int32_t v = 0;
  ASSERT_EQ(bus.read(1, reg::kGoalPosition, &v), feetech::Status::kOk);
  EXPECT_EQ(v, 3331);
}

TEST_F(BusPty, SyncReadReturnsValuesInTheRequestedOrder)
{
  std::vector<int32_t> v;
  ASSERT_EQ(bus.sync_read(reg::kPresentPosition, {3, 1, 2}, &v), feetech::Status::kOk);
  EXPECT_EQ(v, (std::vector<int32_t>{2051, 2049, 2050}));
  EXPECT_EQ(bus.sync_read(reg::kPresentPosition, {1, 9}, &v), feetech::Status::kTimeout);
}

TEST_F(BusPty, LateReplyIsNotTakenForTheNextAnswer)
{
  fake.delay_next_reply_ms = 40;                         // the answer to this read comes too late
  int32_t v = 0;
  EXPECT_EQ(bus.read(1, reg::kModelNumber, &v), feetech::Status::kTimeout);
  std::this_thread::sleep_for(50ms);                     // ... and is now waiting in the input buffer
  ASSERT_EQ(bus.read(2, reg::kPresentPosition, &v), feetech::Status::kOk);
  EXPECT_EQ(v, 2050);                                    // the real answer, not the stale 777
}

TEST_F(BusPty, BlockReadDecodesSeveralRegistersFromOneReply)
{
  // Present_Position (56..57) and Present_Velocity (58..59) of two servos in ONE transaction
  ASSERT_EQ(bus.write(2, reg::kPresentVelocity, -100), feetech::Status::kOk);
  std::vector<feetech::Bytes> data;
  const auto before = bus.stats().transactions;
  ASSERT_EQ(bus.sync_read_block(56, 4, {1, 2}, &data), feetech::Status::kOk);
  EXPECT_EQ(bus.stats().transactions, before + 1);
  ASSERT_EQ(data.size(), 2u);
  ASSERT_EQ(data[1].size(), 4u);
  const uint8_t * d2 = data[1].data();
  EXPECT_EQ(feetech::decode_value(reg::kPresentPosition, d2 + 0), 2050);
  EXPECT_EQ(feetech::decode_value(reg::kPresentVelocity, d2 + 2), -100);   // sign bit 15
  EXPECT_EQ(feetech::decode_value(reg::kPresentPosition, data[0].data()), 2049);
}

TEST_F(BusPty, SecondProgramOnTheSamePortIsRefused)
{
  feetech::FeetechBus second{20ms};
  std::string error;
  EXPECT_FALSE(second.open(fake.device(), 1000000, &error));
  EXPECT_NE(error.find("in use by another program"), std::string::npos) << error;
  bus.close();                                          // first one lets go ...
  EXPECT_TRUE(second.open(fake.device(), 1000000, &error)) << error;   // ... now it works
}
