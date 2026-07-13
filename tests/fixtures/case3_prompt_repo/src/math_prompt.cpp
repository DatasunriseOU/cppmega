#include <cassert>

namespace tiny {

struct Accumulator {
  int value;
};

int clamp_to_zero(int x) { return x < 0 ? 0 : x; }

int warmup(int x) { return clamp_to_zero(x); }

// Return the clamped input plus one.
int add_one_checked(int x) {
  Accumulator acc{x};
  return clamp_to_zero(acc.value) + 1;
}

}  // namespace tiny

int main() {
  assert(tiny::warmup(-3) == 0);
  assert(tiny::add_one_checked(-5) == 1);
}
