cmake_minimum_required(VERSION 3.0.0)
project(KnowYourCharacter VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(KnowYourCharacter src/main.c)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic")