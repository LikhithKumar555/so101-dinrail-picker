// Unit tests for the safety rules - no ROS needed, runs in milliseconds.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "so101_safety/safety_logic.hpp"

using so101_safety::Level;
using so101_safety::Limits;
using so101_safety::SafetyLogic;
using so101_safety::ServoReading;

namespace
{
std::vector<ServoReading> one(double load, double temp, double volt = 12.0)
{
  return {ServoReading{"elbow_flex", load, temp, volt}};
}
}  // namespace

TEST(SafetyLogic, NominalIsOk)
{
  SafetyLogic s{Limits{}};
  s.update(one(0.2, 35.0), 0.0);
  EXPECT_EQ(s.level(), Level::OK);
}

TEST(SafetyLogic, TemperatureWarnThenFault)
{
  SafetyLogic s{Limits{}};
  s.update(one(0.2, 58.0), 0.0);
  EXPECT_EQ(s.level(), Level::WARN);
  EXPECT_FALSE(s.faulted());
  s.update(one(0.2, 66.0), 0.1);
  EXPECT_EQ(s.level(), Level::FAULT);
  EXPECT_NE(s.fault_reason().find("temperature"), std::string::npos);
}

TEST(SafetyLogic, ShortLoadSpikeIsOnlyAWarning)
{
  SafetyLogic s{Limits{}};  // load_fault 0.85 must last 0.5 s
  s.update(one(0.95, 35.0), 0.0);
  s.update(one(0.95, 35.0), 0.3);
  EXPECT_EQ(s.level(), Level::WARN);
  s.update(one(0.2, 35.0), 0.4);  // spike over
  s.update(one(0.95, 35.0), 0.5);  // new spike: timer restarts
  s.update(one(0.95, 35.0), 0.8);
  EXPECT_FALSE(s.faulted());
}

TEST(SafetyLogic, SustainedOverloadFaults)
{
  SafetyLogic s{Limits{}};
  s.update(one(-0.9, 35.0), 0.0);  // sign does not matter
  s.update(one(-0.9, 35.0), 0.6);
  EXPECT_TRUE(s.faulted());
  EXPECT_NE(s.fault_reason().find("load"), std::string::npos);
}

TEST(SafetyLogic, UndervoltageFaults)
{
  SafetyLogic s{Limits{}};
  s.update(one(0.2, 35.0, 5.0), 0.0);
  EXPECT_TRUE(s.faulted());
}

TEST(SafetyLogic, FaultIsLatchedUntilReset)
{
  SafetyLogic s{Limits{}};
  s.update(one(0.2, 70.0), 0.0);
  s.update(one(0.2, 40.0), 0.1);  // cause is gone ...
  EXPECT_EQ(s.level(), Level::FAULT);  // ... but the fault stays latched
  std::string why;
  EXPECT_TRUE(s.reset(0.2, why)) << why;
  EXPECT_EQ(s.level(), Level::OK);
}

TEST(SafetyLogic, ResetRefusedWhileCauseIsPresent)
{
  SafetyLogic s{Limits{}};
  s.update(one(0.2, 70.0), 0.0);
  std::string why;
  EXPECT_FALSE(s.reset(0.1, why));
  EXPECT_TRUE(s.faulted());
}

TEST(SafetyLogic, TelemetryTimeoutFaults)
{
  SafetyLogic s{Limits{}};
  s.check_timeout(0.0);
  s.update(one(0.2, 35.0), 0.1);
  s.check_timeout(0.5);
  EXPECT_FALSE(s.faulted());
  s.check_timeout(0.7);  // 0.6 s without data > 0.5 s
  EXPECT_TRUE(s.faulted());
  std::string why;
  EXPECT_FALSE(s.reset(0.8, why));  // still no fresh data
}

TEST(SafetyLogic, NoTelemetryAtStartupFaultsAfterGracePeriod)
{
  SafetyLogic s{Limits{}};
  s.check_timeout(0.0);
  s.check_timeout(4.0);
  EXPECT_FALSE(s.faulted());
  s.check_timeout(5.5);
  EXPECT_TRUE(s.faulted());
}
