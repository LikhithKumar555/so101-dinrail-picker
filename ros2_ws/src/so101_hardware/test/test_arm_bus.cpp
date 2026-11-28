// ArmBus against fake servos on a pseudo-terminal.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>

#include "fake_servo_bus.hpp"
#include "so101_hardware/arm_bus.hpp"

using namespace std::chrono_literals;
using so101_hardware::ArmBus;
using so101_hardware::JointConfig;
using so101_hardware::kRadPerStep;
using so101_hardware::ServoState;
using so101_hardware_test::FakeServoBus;
namespace reg = feetech::reg;

namespace
{
std::vector<JointConfig> three_joints()
{
  std::vector<JointConfig> j(3);
  j[0] = {"a", 1, 2048, 1, -1.5, 1.5};
  j[1] = {"b", 2, 2048, -1, -1.5, 1.5};     // mounted mirrored
  j[2] = {"c", 3, 1000, 1, -1.5, 1.5};      // other calibration
  return j;
}

class ArmBusTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::string error;
    ASSERT_TRUE(arm.open(fake.device(), 1000000, &error)) << error;
  }
  FakeServoBus fake{{1, 2, 3}};
  ArmBus arm{three_joints(), 10ms};
};
}  // namespace

TEST_F(ArmBusTest, ServoCheckNamesTheMissingServo)
{
  std::string error;
  EXPECT_TRUE(arm.check_servos(&error)) << error;
  fake.drop_replies(1);                     // one lost reply is retried, not reported
  EXPECT_TRUE(arm.check_servos(&error)) << error;
  fake.unplug(2);
  EXPECT_FALSE(arm.check_servos(&error));
  EXPECT_NE(error.find("b (id 2): timeout"), std::string::npos) << error;
  fake.set(3, reg::kModelNumber, 1234);
  EXPECT_FALSE(arm.check_servos(&error));
  EXPECT_NE(error.find("not an STS3215"), std::string::npos) << error;
}

TEST_F(ArmBusTest, OneTransactionReadsTheWholeState)
{
  fake.set(1, reg::kPresentPosition, 2048 + 100);
  fake.set(1, reg::kPresentVelocity, -50);
  fake.set(1, reg::kPresentLoad, -250);              // 25 % duty, negative direction
  fake.set(2, reg::kPresentPosition, 2048 + 100);    // mirrored joint: same reading, other sign
  fake.set(2, reg::kPresentLoad, 400);
  fake.set(3, reg::kPresentTemperature, 58);
  fake.set_error_byte(3, feetech::err::kOverheat);

  std::vector<ServoState> s;
  uint8_t bits = 0;
  const auto before = arm.stats().transactions;
  ASSERT_EQ(arm.read_state(&s, &bits), feetech::Status::kOk);
  EXPECT_EQ(arm.stats().transactions, before + 1);
  ASSERT_EQ(s.size(), 3u);
  EXPECT_NEAR(s[0].position, 100 * kRadPerStep, 1e-12);
  EXPECT_NEAR(s[0].velocity, -50 * kRadPerStep, 1e-12);
  EXPECT_DOUBLE_EQ(s[0].load, -0.25);
  EXPECT_NEAR(s[1].position, -100 * kRadPerStep, 1e-12);
  EXPECT_DOUBLE_EQ(s[1].load, -0.40);
  EXPECT_NEAR(s[2].position, (2048 - 1000) * kRadPerStep, 1e-12);
  EXPECT_DOUBLE_EQ(s[0].voltage, 7.4);
  EXPECT_DOUBLE_EQ(s[2].temperature, 58.0);
  EXPECT_EQ(bits, feetech::err::kOverheat);
  EXPECT_EQ(ArmBus::describe_errors(bits), "overheat");
}

TEST_F(ArmBusTest, WithoutSafeStartTheServoJumpsToItsOldGoal)
{
  // what goes wrong without hold_current_pose(): the arm was moved by hand while torque was off
  fake.set(1, reg::kGoalPosition, 1200);             // old goal, still in the servo's RAM
  fake.set(1, reg::kPresentPosition, 2300);          // where the joint really is
  ASSERT_EQ(arm.set_torque(true), feetech::Status::kOk);
  EXPECT_EQ(fake.get(1, reg::kPresentPosition), 1200);   // 1100 steps = 97 deg jump
}

TEST_F(ArmBusTest, SafeStartHoldsTheCurrentPoseEvenBeyondTheLimits)
{
  fake.set(1, reg::kGoalPosition, 1200);
  fake.set(1, reg::kPresentPosition, 2300);
  fake.set(2, reg::kPresentPosition, 2048 + 1200);   // 1.84 rad: beyond the 1.5 rad soft limit
  std::vector<ServoState> s;
  std::string error;
  ASSERT_EQ(arm.hold_current_pose(&s, &error), feetech::Status::kOk) << error;
  for (uint8_t id : {1, 2, 3}) {
    EXPECT_EQ(fake.get(id, reg::kTorqueEnable), 1);
  }
  EXPECT_EQ(fake.get(1, reg::kPresentPosition), 2300);   // no jump
  EXPECT_EQ(fake.get(2, reg::kPresentPosition), 2048 + 1200);   // not clamped to the limit
  EXPECT_NEAR(s[0].position, (2300 - 2048) * kRadPerStep, 1e-12);
}

TEST_F(ArmBusTest, SafeStartRefusesWhenTheGoalsDidNotArrive)
{
  // (if the old goal happened to equal the position, a lost write would do no harm - and the
  //  read-back could not tell the difference. So: a goal that differs.)
  fake.set(1, reg::kGoalPosition, 1200);
  fake.set(1, reg::kPresentPosition, 2300);
  fake.ignore_sync_writes(true);                      // the broadcast is lost on the wire
  std::vector<ServoState> s;
  std::string error;
  EXPECT_EQ(arm.hold_current_pose(&s, &error), feetech::Status::kBadReply);
  EXPECT_NE(error.find("Goal_Position did not arrive"), std::string::npos) << error;
  EXPECT_EQ(fake.get(1, reg::kTorqueEnable), 0);      // torque was never switched on
}

TEST_F(ArmBusTest, SafeStartSurvivesALostReplyInEveryStep)
{
  // what happened on the laptop: one late reply during start-up. Each setup step retries.
  fake.set(1, reg::kGoalPosition, 1200);
  fake.set(1, reg::kPresentPosition, 2300);
  std::vector<ServoState> s;
  std::string error;
  fake.drop_replies(1);                                // the first state read gets no reply
  ASSERT_EQ(arm.hold_current_pose(&s, &error), feetech::Status::kOk) << error;
  EXPECT_EQ(arm.setup_retries_used(), 1);
  EXPECT_EQ(fake.get(1, reg::kPresentPosition), 2300);   // still no jump
  EXPECT_EQ(fake.get(1, reg::kTorqueEnable), 1);
}

TEST_F(ArmBusTest, SafeStartGivesUpAfterTheLastAttempt)
{
  fake.drop_replies(100);                              // the bus is really broken
  std::vector<ServoState> s;
  std::string error;
  EXPECT_EQ(arm.hold_current_pose(&s, &error), feetech::Status::kTimeout);
  EXPECT_EQ(arm.setup_retries_used(), 3);              // 3 attempts, then stop
  fake.drop_replies(0);
  EXPECT_EQ(fake.get(1, reg::kTorqueEnable), 0);
}

TEST_F(ArmBusTest, ControlLoopReadsAreNotRetried)
{
  fake.drop_replies(1);
  std::vector<ServoState> s;
  EXPECT_EQ(arm.read_state(&s), feetech::Status::kTimeout);   // reported at once
  EXPECT_EQ(arm.setup_retries_used(), 0);
  EXPECT_EQ(arm.read_state(&s), feetech::Status::kOk);        // next cycle is fine again
}

TEST_F(ArmBusTest, GoalsAreClampedAndNanLeavesAServoAlone)
{
  ASSERT_EQ(arm.set_torque(true), feetech::Status::kOk);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  ASSERT_EQ(arm.write_goals({0.5, 9.0, nan}), feetech::Status::kOk);
  std::this_thread::sleep_for(5ms);                   // SYNC_WRITE has no reply: give it time
  EXPECT_EQ(fake.get(1, reg::kGoalPosition), 2048 + std::lround(0.5 / kRadPerStep));
  EXPECT_EQ(fake.get(2, reg::kGoalPosition), 2048 - std::lround(1.5 / kRadPerStep));  // clamped
  EXPECT_EQ(fake.get(3, reg::kGoalPosition), 2048);   // unchanged
}

TEST_F(ArmBusTest, AnUnpluggedServoMakesTheReadTimeOut)
{
  fake.unplug(3);
  std::vector<ServoState> s;
  EXPECT_EQ(arm.read_state(&s), feetech::Status::kTimeout);
  EXPECT_EQ(arm.stats().timeouts, 1u);
}
