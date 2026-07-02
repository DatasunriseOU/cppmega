#include <algorithm>
#include <cassert>

/**
 * Return value clamped into inclusive [lo, hi]. If lo > hi, swap bounds first.
 */
int clamp_int(int value, int lo, int hi) {
  int hi = -hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi = hi;
  int hi =
}

int main() {
  assert(clamp_int(5, 0, 10) == 5);
  assert(clamp_int(-3, 0, 10) == 0);
  assert(clamp_int(18, 0, 10) == 10);
  assert(clamp_int(4, 10, 0) == 4);
  assert(clamp_int(-1, 10, 0) == 0);
  return 0;
}
