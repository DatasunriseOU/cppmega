#!/bin/bash
# megacpp data prep — Stage 1: Download raw C/C++ source corpora.
#
# Adopted from nanochat/scripts/data/download_cpp_sources.sh.
# Clones a fixed set of C/C++ repositories used as the megacpp training corpus
# foundation. No credentials required — all sources are public GitHub repos.
#
# Sibling stages: see scripts/data/prepare_data.sh.
#
# Usage:
#   bash prepare_download_megacpp.sh [DATA_DIR]
#
# Default DATA_DIR: ${MEGACPP_DATA_ROOT:-/home/dave/cppmega-root/data}/cpp_raw
#
# Output: ${DATA_DIR} containing shallow clones of each repo.
set -euo pipefail

DEFAULT_ROOT="${MEGACPP_DATA_ROOT:-/home/dave/cppmega-root/data}"
DATA_DIR="${1:-${DEFAULT_ROOT}/cpp_raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== megacpp: downloading C/C++ sources to $DATA_DIR ==="

clone_shallow() {
    local name="$1"
    local url="$2"
    local ref="${3:-}"
    if [ -d "$name" ]; then
        echo "--- $name already exists, skipping ---"
        return 0
    fi
    echo "--- Cloning $name ---"
    if [ -n "$ref" ]; then
        git clone --depth=1 --branch="$ref" "$url" "$name"
    else
        git clone --depth=1 "$url" "$name"
    fi
}

# LLVM/Clang — modern C++
clone_shallow llvm-project https://github.com/llvm/llvm-project.git llvmorg-19.1.0
# Boost — advanced template C++
if [ ! -d "boost" ]; then
    echo "--- Cloning Boost 1.86.0 (shallow + submodules) ---"
    git clone --depth=1 --branch=boost-1.86.0 --recurse-submodules --shallow-submodules \
        https://github.com/boostorg/boost.git
fi
# Linux kernel — C, systems programming
clone_shallow linux https://github.com/torvalds/linux.git v6.10
clone_shallow fmt https://github.com/fmtlib/fmt.git 11.0.0
clone_shallow googletest https://github.com/google/googletest.git v1.15.0
clone_shallow abseil-cpp https://github.com/abseil/abseil-cpp.git
clone_shallow folly https://github.com/facebook/folly.git
clone_shallow grpc https://github.com/grpc/grpc.git v1.67.0

echo ""
echo "=== Downloads complete ==="
echo "Counting C/C++ files..."
find "$DATA_DIR" -type f \( \
    -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \
    -o -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.hxx" \
    \) | wc -l
echo "Total size:"
du -sh "$DATA_DIR"
