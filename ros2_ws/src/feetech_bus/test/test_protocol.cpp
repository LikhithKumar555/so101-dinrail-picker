// Same test vectors as the Python reference (sim/tests/test_feetech_protocol.py).
#include <gtest/gtest.h>

#include <stdexcept>

#include "feetech_bus/protocol.hpp"

using feetech::Bytes;
namespace reg = feetech::reg;

TEST(Protocol, PingMatchesTheDocumentedBytes)
{
  // checksum = ~(1 + 2 + 1) & 0xFF = 0xFB
  EXPECT_EQ(feetech::ping(1), (Bytes{0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB}));
}

TEST(Protocol, WriteIsLittleEndian)
{
  const Bytes p = feetech::write(1, reg::kGoalPosition, 2048);   // 0x0800 -> 00 08
  EXPECT_EQ(Bytes(p.begin(), p.begin() + 8), (Bytes{0xFF, 0xFF, 1, 5, feetech::kWrite, 42, 0x00, 0x08}));
  EXPECT_EQ(p.back(), feetech::checksum(p.data() + 2, p.size() - 3));
}

TEST(Protocol, SignMagnitudeIsNotTwosComplement)
{
  EXPECT_EQ(feetech::encode_value(reg::kPresentVelocity, -5), (Bytes{0x05, 0x80}));
  const Bytes raw{0x05, 0x80};
  EXPECT_EQ(feetech::decode_value(reg::kPresentVelocity, raw.data()), -5);
  const uint16_t load = 1024 + 300;                                 // sign bit 10
  const Bytes lb{static_cast<uint8_t>(load & 0xFF), static_cast<uint8_t>(load >> 8)};
  EXPECT_EQ(feetech::decode_value(reg::kPresentLoad, lb.data()), -300);
  EXPECT_THROW(feetech::encode_value(reg::kHomingOffset, 3000), std::out_of_range);
}

TEST(Protocol, SyncWriteLayout)
{
  const Bytes p = feetech::sync_write(reg::kGoalPosition, {{1, 100}, {2, 200}});
  EXPECT_EQ(p[2], feetech::kBroadcastId);
  EXPECT_EQ(p[4], feetech::kSyncWrite);
  EXPECT_EQ(Bytes(p.begin() + 5, p.begin() + 13), (Bytes{42, 2, 1, 100, 0, 2, 200, 0}));
  EXPECT_EQ(p[3], p.size() - 4);                                    // LEN: from INSTR to checksum
}

TEST(Protocol, ParserHandlesSplitGluedAndCorruptedStreams)
{
  Bytes stream{0x00, 0x13};
  const Bytes a = feetech::ping(1), b = feetech::read(2, reg::kPresentPosition);
  stream.insert(stream.end(), a.begin(), a.end());
  stream.insert(stream.end(), b.begin(), b.end());
  feetech::PacketParser parser;
  std::vector<feetech::Packet> got;
  for (std::size_t i = 0; i < stream.size(); i += 3) {               // 3-byte chunks
    for (auto & p : parser.feed(stream.data() + i, std::min<std::size_t>(3, stream.size() - i))) {
      got.push_back(p);
    }
  }
  ASSERT_EQ(got.size(), 2u);
  EXPECT_EQ(got[0].id, 1);
  EXPECT_EQ(got[0].code, feetech::kPing);
  EXPECT_EQ(got[1].id, 2);
  EXPECT_EQ(got[1].params, (Bytes{56, 2}));

  Bytes bad = feetech::ping(3);
  bad.back() ^= 0xFF;
  const Bytes good = feetech::ping(4);
  bad.insert(bad.end(), good.begin(), good.end());
  feetech::PacketParser p2;
  const auto r = p2.feed(bad.data(), bad.size());
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].id, 4);
  EXPECT_EQ(p2.bad_checksums(), 1u);
}

TEST(Protocol, RegisterLookupByName)
{
  ASSERT_NE(feetech::find_register("Present_Position"), nullptr);
  EXPECT_EQ(feetech::find_register("Present_Position")->address, 56);
  EXPECT_EQ(feetech::find_register("No_Such_Register"), nullptr);
}

TEST(Protocol, WireTime)
{
  EXPECT_DOUBLE_EQ(feetech::transfer_time_s(10, 1000000), 100e-6);
}
