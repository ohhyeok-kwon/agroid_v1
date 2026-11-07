cmake_minimum_required(VERSION 3.16)
project(humanoid_controller_msg)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_generator_rs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotorCmd.msg" "msg/MotorState.msg" "msg/IMUState.msg"
  "msg/LowCmd.msg" "msg/LowState.msg")
ament_export_dependencies(rosidl_default_runtime)
ament_package()
