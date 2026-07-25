#pragma once

namespace provider {
struct Widget {
    int value;
    int size() const { return value; }
};
}
