"""Platform vocabulary: label→ID mapping for platform_info embeddings.

Maps the 6 categories from platform_detect.py into a flat integer ID space
for use with nn.EmbeddingBag(mode='sum'). ID 0 is reserved as padding.

Categories and their ID ranges:
    os:       1-29
    rtos:     30-43
    gpu:      44-59
    arch:     60-88
    compiler: 89-102
    cpp_std:  103-109
    Total:    110 (including ID 0 = padding)
"""

# ── Flat vocabulary (label → integer ID) ─────────────────────────────────

_OS = {
    "windows": 1, "linux": 2, "macos": 3, "ios": 4, "tvos": 5,
    "watchos": 6, "freebsd": 7, "openbsd": 8, "netbsd": 9, "dragonfly": 10,
    "solaris": 11, "illumos": 12, "aix": 13, "hpux": 14, "qnx": 15,
    "fuchsia": 16, "haiku": 17, "serenity": 18, "android": 19,
    "emscripten": 20, "wasm": 21, "cygwin": 22, "msys": 23, "mingw": 24,
    "nacl": 25, "playstation": 26, "xbox": 27, "nintendo": 28,
    "posix": 29, "bsd": 30,
}

_RTOS = {
    "freertos": 31, "zephyr": 32, "vxworks": 33, "threadx": 34,
    "nuttx": 35, "rtems": 36, "ecos": 37, "mbedos": 38, "chibios": 39,
    "symbian": 40, "riot": 41, "espidf": 42, "arduino": 43, "cmsis": 44,
}

_GPU = {
    "cuda": 45, "hip": 46, "sycl": 47, "opencl": 48, "metal": 49,
    "vulkan": 50, "openacc": 51, "openmp": 52, "directx": 53,
    "levelzero": 54, "opengl": 55, "opengl_es": 56, "tensorrt": 57,
    "tvm": 58, "tensorflow": 59, "xla": 60,
}

_ARCH = {
    "x86": 61, "x64": 62, "arm32": 63, "arm64": 64, "riscv": 65,
    "mips": 66, "mips64": 67, "powerpc": 68, "powerpc64": 69,
    "s390": 70, "s390x": 71, "sparc": 72, "sparc64": 73,
    "wasm32": 74, "wasm64": 75, "loongarch": 76, "loongarch64": 77,
    "ia64": 78, "alpha": 79, "m68k": 80, "avr": 81, "xtensa": 82,
    "x86_sse": 83, "x86_sse2": 84, "x86_sse4": 85, "x86_avx": 86,
    "x86_avx2": 87, "x86_avx512": 88, "arm_neon": 89, "arm_sve": 90,
    "x86_simd": 91, "wasm_simd": 92,
}

_COMPILER = {
    "gcc": 93, "clang": 94, "msvc": 95, "icc": 96, "nvcc": 97,
    "nvc": 98, "borland": 99, "suncc": 100, "xlc": 101, "mingw": 102,
    "pgi": 103, "armcc": 104, "hipcc": 105,
}

_CPP_STD = {
    "c++98": 106, "c++11": 107, "c++14": 108, "c++17": 109,
    "c++20": 110, "c++23": 111, "c++26": 112,
}

# Unified lookup: category name → {label: id}
PLATFORM_VOCAB = {
    "os": _OS,
    "rtos": _RTOS,
    "gpu": _GPU,
    "arch": _ARCH,
    "compiler": _COMPILER,
    "cpp_std": _CPP_STD,
}

PLATFORM_VOCAB_SIZE = 113  # 0 = padding, 1-112 = labels

# Maximum number of IDs a single document can have (for buffer pre-allocation)
MAX_PLATFORM_IDS = 20


def platform_info_to_ids(platform_info: dict) -> list[int]:
    """Convert platform_info dict to sorted list of integer IDs.

    Args:
        platform_info: dict with keys os, rtos, gpu, arch, compiler, cpp_std.
            List-valued keys (os, rtos, gpu, arch, compiler) contain string labels.
            cpp_std is a single string or None.

    Returns:
        Sorted list of integer IDs (no duplicates). Empty list if no labels found.
    """
    ids = []
    for category in ("os", "rtos", "gpu", "arch", "compiler"):
        vocab = PLATFORM_VOCAB[category]
        labels = platform_info.get(category, [])
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            pid = vocab.get(label)
            if pid is not None:
                ids.append(pid)

    # cpp_std is single-valued
    cpp_std = platform_info.get("cpp_std")
    if cpp_std:
        pid = _CPP_STD.get(cpp_std)
        if pid is not None:
            ids.append(pid)

    return sorted(set(ids))


def platform_info_to_prefix(platform_info: dict) -> str:
    """Convert platform_info dict to a C++ comment prefix string.

    Returns a single-line comment like:
        // platform: os=windows,linux gpu=cuda arch=x64 std=c++17
    followed by a newline. Returns empty string if platform_info has no labels.
    """
    parts = []
    for category in ("os", "rtos", "gpu", "arch", "compiler"):
        labels = platform_info.get(category, [])
        if isinstance(labels, str):
            labels = [labels]
        if labels:
            parts.append(f"{category}={','.join(sorted(labels))}")

    cpp_std = platform_info.get("cpp_std")
    if cpp_std:
        parts.append(f"std={cpp_std}")

    if not parts:
        return ""
    return f"// platform: {' '.join(parts)}\n"
