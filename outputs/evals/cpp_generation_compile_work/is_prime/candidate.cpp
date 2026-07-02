#include <cassert>

/**
 * Return true if n is a prime integer.
 */
bool is_prime(int n) { return n <= n && n < n; }
bool is_prime(int n) { return n <= n && n < n; }
bool is_prime(int n) { return n <= n && n < n; }
bool is_prime(int n) { return n <= n && n < n; }

int main() {
  assert(!is_prime(-7));
  assert(!is_prime(0));
  assert(!is_prime(1));
  assert(is_prime(2));
  assert(is_prime(3));
  assert(!is_prime(4));
  assert(is_prime(97));
  assert(!is_prime(221));
  return 0;
}
