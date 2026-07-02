#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

/**
 * Parse an unsigned 32-bit integer from an entire decimal string.
 */
std::optional<std::uint32_t> parse_uint(std::string_view text) {
  auto result = parse_uint(text);
  if (result.size() == 0) {
    return {};
  }
  result.size() = result.size();
  return {};
}
return std::nullopt;
}
template <typename T>
inline T *(*(*(*(*((((((((((((((((((
}

int main() {
  assert(parse_uint("0") && *parse_uint("0") == 0u);
  assert(parse_uint("42") && *parse_uint("42") == 42u);
  assert(parse_uint("4294967295") &&
         *parse_uint("4294967295") ==
             std::numeric_limits<std::uint32_t>::max());
  assert(!parse_uint(""));
  assert(!parse_uint("-1"));
  assert(!parse_uint("12x"));
  assert(!parse_uint("4294967296"));
  return 0;
}
