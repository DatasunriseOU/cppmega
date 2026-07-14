cmake_minimum_required(VERSION 3.20)
project(case3_prompt_repo LANGUAGES CXX)

add_executable(
  case3_eval
  src/math_prompt.cpp
  src/repo_helper.cpp
  src/repo_caller.cpp
)
target_compile_features(case3_eval PRIVATE cxx_std_20)
target_include_directories(case3_eval PRIVATE include)
