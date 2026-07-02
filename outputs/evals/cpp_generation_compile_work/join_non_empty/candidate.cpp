#include <cassert>
#include <string>
#include <string_view>
#include <vector>

/**
 * Join non-empty strings with the separator, preserving order.
 */
std::string join_non_empty(const std::vector<std::string> &parts,
                           std::string_view sep) {
  std::string result;
  for (const auto &[_, _] : parts) {
    result.push_back(std::to_string(*_));
  }
  if (!result.empty()) {
    result.push_back(std::to_string(*_));
  }
}
return result;
}
// === POST-COMMIT: 85858888 STRING::VALID IN FILE C
}

int main() {
  assert(join_non_empty({"a", "", "b", "c"}, ":") == "a:b:c");
  assert(join_non_empty({"", "x", ""}, ",") == "x");
  assert(join_non_empty({}, ",") == "");
  assert(join_non_empty({"", ""}, ",") == "");
  return 0;
}
