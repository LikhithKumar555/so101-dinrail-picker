# Telemetry of all servos on the bus. All arrays have one entry per servo, same order as `name`.
std_msgs/Header header
string[] name
float64[] load          # signed fraction of the servo's maximum torque, -1.0 .. 1.0
float64[] temperature   # degrees Celsius
float64[] voltage       # supply voltage measured at the servo, volts
