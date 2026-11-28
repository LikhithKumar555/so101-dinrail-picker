// The plugin as the controller manager sees it: the ResourceManager parses <ros2_control> from
// a URDF, loads so101_hardware/FeetechSystem with pluginlib and drives its lifecycle and
// read()/write(). The servos are the pty fakes from fake_servo_bus.hpp.
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "fake_servo_bus.hpp"
#include "hardware_interface/resource_manager.hpp"
#include "hardware_interface/types/lifecycle_state_names.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "so101_hardware/joint_codec.hpp"

using namespace std::chrono_literals;
using hardware_interface::return_type;
using lifecycle_msgs::msg::State;
using so101_hardware::kRadPerStep;
using so101_hardware_test::FakeServoBus;
namespace reg = feetech::reg;

namespace
{
std::string urdf(const std::string & device, const std::string & j2_params = "<param name=\"id\">2</param>")
{
  return R"(<?xml version="1.0"?>
<robot name="two_joints">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5" effort="1" velocity="5"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5" effort="1" velocity="5"/></joint>
  <ros2_control name="so101_system" type="system">
    <hardware>
      <plugin>so101_hardware/FeetechSystem</plugin>
      <param name="device">)" + device + R"(</param>
      <param name="timeout_ms">5</param>
      <param name="max_consecutive_errors">3</param>
    </hardware>
    <joint name="j1">
      <param name="id">1</param>
      <command_interface name="position"><param name="min">-1.5</param><param name="max">1.5</param></command_interface>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
    <joint name="j2">
      )" + j2_params + R"(
      <command_interface name="position"><param name="min">-1.5</param><param name="max">1.5</param></command_interface>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
  </ros2_control>
</robot>)";
}

// move the component to a lifecycle state (the controller manager does the same at start-up)
return_type go_to(hardware_interface::ResourceManager & rm, uint8_t id, const std::string & label)
{
  rclcpp_lifecycle::State target(id, label);
  return rm.set_component_state("so101_system", target);
}
return_type configure(hardware_interface::ResourceManager & rm)
{
  return go_to(rm, State::PRIMARY_STATE_INACTIVE, hardware_interface::lifecycle_state_names::INACTIVE);
}
return_type activate(hardware_interface::ResourceManager & rm)
{
  return go_to(rm, State::PRIMARY_STATE_ACTIVE, hardware_interface::lifecycle_state_names::ACTIVE);
}

class FeetechSystemTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  std::unique_ptr<hardware_interface::ResourceManager> make(const std::string & description)
  {
    return std::make_unique<hardware_interface::ResourceManager>(
      description, node->get_node_clock_interface(), node->get_node_logging_interface(), false, 100);
  }
  return_type cycle(hardware_interface::ResourceManager & rm)
  {
    const auto t = node->now();
    const auto r = rm.read(t, rclcpp::Duration(10ms)).result;
    if (r != return_type::OK) {return r;}
    return rm.write(t, rclcpp::Duration(10ms)).result;
  }

  FakeServoBus fake{{1, 2}};
  rclcpp::Node::SharedPtr node = std::make_shared<rclcpp::Node>("test_feetech_system");
};
}  // namespace

TEST_F(FeetechSystemTest, ConfigureReadsThePoseActivateHoldsItWriteMovesIt)
{
  fake.set(1, reg::kPresentPosition, 2048 + 300);
  fake.set(1, reg::kGoalPosition, 100);                 // stale goal in RAM
  auto rm = make(urdf(fake.device()));
  ASSERT_TRUE(rm->are_components_initialized());

  // configure: port open, servos checked, real pose in the state interfaces, torque still off
  ASSERT_EQ(configure(*rm),
    return_type::OK);
  auto q1 = rm->claim_state_interface("j1/position");
  EXPECT_NEAR(q1.get_optional().value(), 300 * kRadPerStep, 1e-9);
  EXPECT_EQ(fake.get(1, reg::kTorqueEnable), 0);

  // activate: safe start
  ASSERT_EQ(activate(*rm),
    return_type::OK);
  EXPECT_EQ(fake.get(1, reg::kTorqueEnable), 1);
  EXPECT_EQ(fake.get(1, reg::kPresentPosition), 2048 + 300);     // did not jump to the stale goal

  // a controller commands j2 = 0.4 rad; j1 keeps the command it got on activation (= hold)
  auto cmd2 = rm->claim_command_interface("j2/position");
  ASSERT_TRUE(cmd2.set_value(0.4));
  ASSERT_EQ(cycle(*rm), return_type::OK);
  std::this_thread::sleep_for(5ms);                      // SYNC_WRITE has no reply
  EXPECT_EQ(fake.get(2, reg::kGoalPosition), 2048 + std::lround(0.4 / kRadPerStep));
  EXPECT_EQ(fake.get(1, reg::kGoalPosition), 2048 + 300);
  ASSERT_EQ(cycle(*rm), return_type::OK);                // the fake is at its goal at once
  auto q2 = rm->claim_state_interface("j2/position");
  EXPECT_NEAR(q2.get_optional().value(), 0.4, 0.5 * kRadPerStep);
}

TEST_F(FeetechSystemTest, LostRepliesAreToleratedUntilTheLimitThenTheHardwareFails)
{
  auto rm = make(urdf(fake.device()));
  ASSERT_EQ(activate(*rm),
    return_type::OK);
  ASSERT_EQ(cycle(*rm), return_type::OK);
  fake.unplug(2);
  EXPECT_EQ(cycle(*rm), return_type::OK);                // 1st and 2nd miss: keep going
  EXPECT_EQ(cycle(*rm), return_type::OK);
  EXPECT_EQ(cycle(*rm), return_type::ERROR);             // 3rd in a row: give up
}

TEST_F(FeetechSystemTest, MissingServoIdIsRejectedAtInit)
{
  auto rm = make(urdf(fake.device(), ""));
  EXPECT_FALSE(rm->are_components_initialized());
}

TEST_F(FeetechSystemTest, MissingServoFailsConfigure)
{
  fake.unplug(1);
  auto rm = make(urdf(fake.device()));
  ASSERT_TRUE(rm->are_components_initialized());
  EXPECT_EQ(configure(*rm),
    return_type::ERROR);
}
