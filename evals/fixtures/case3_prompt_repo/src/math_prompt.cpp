#include "repo_api.hpp"

namespace case3_repo {

// Return the clamped input plus one.
int add_one_checked(int x) {
  Accumulator acc{x};
  const int clamped = repository_helper(acc.value);
  return clamped + 1;
}

} // namespace case3_repo

int main() {
  if (case3_repo::repository_caller(-3) != 1)
    return 1;
  if (case3_repo::add_one_checked(-5) != 1)
    return 2;
  if (case3_repo::add_one_checked(4) != 5)
    return 3;
  return 0;
}
