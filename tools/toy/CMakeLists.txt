cmake_minimum_required(VERSION 3.20.0)
project(standalone-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

add_custom_target(Toy)
set_target_properties(Toy PROPERTIES FOLDER Examples)

macro(add_toy_chapter name)
  add_dependencies(Toy ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)

add_subdirectory(Ch1)
add_subdirectory(Ch2)
add_subdirectory(Ch3)
add_subdirectory(Ch4)
add_subdirectory(Ch5)
add_subdirectory(Ch6)
add_subdirectory(Ch7)
