#include "repo_api.hpp"

namespace case3_repo {

Accumulator::Accumulator(int initial) : value(initial) {}

int repository_helper(int value) { return value < 0 ? 0 : value; }
double repository_helper(double value) { return value + 0.25; }

} // namespace case3_repo
