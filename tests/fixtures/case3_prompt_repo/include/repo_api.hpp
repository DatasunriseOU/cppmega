#pragma once

namespace case3_repo {

struct Accumulator {
  explicit Accumulator(int initial);

  int value;
};

int repository_helper(int value);
double repository_helper(double value);
int repository_caller(int value);
int add_one_checked(int value);

} // namespace case3_repo
