#include <stdint.h>

#include <array>
#include <memory>
#include <atomic>
#include <mutex>

// This is an array of exponentially growing chunks
// Only addition of elements needs to be protected
// by a lock while elements remain accessible even
// during a resizing operation

template<typename T>
class ConcurrentArray {
  static const uint64_t offset = 8u;

  std::atomic<uint64_t> elements{0};
  std::array<std::unique_ptr<T[]>, 64> blocks{};
  mutable std::mutex mutex;

  static uint64_t msb(uint64_t x) {
    return UINT64_C(63) - __builtin_clzll(x);
  }
  static uint64_t block(uint64_t i) {
    return msb((i>>(offset-1u))|1u);
  }
  static uint64_t index(uint64_t i, uint64_t b) {
    uint64_t saturated = b?b-1:0;
    uint64_t block_size = (1u<<(saturated+offset));
    uint64_t mask = block_size - 1u;
    return i & mask;
  }
public:

  uint64_t size() const {
    return elements;
  }
  T& operator[](uint64_t i) {
    uint64_t b = block(i);
    uint64_t idx = index(i, b);
    return blocks[b][idx];
  }
  const T& operator[](uint64_t i) const {
    uint64_t b = block(i);
    uint64_t idx = index(i, b);
    return blocks[b][idx];
  }
  uint64_t add() {
    uint64_t index = elements++;
    uint64_t b = block(index);

    std::lock_guard<std::mutex> guard(mutex);
    if(blocks[b] == nullptr) {
      uint64_t saturated = b?b-1:0;
      uint64_t block_size = (1u<<(saturated+offset));
      blocks[b].reset(new T[block_size]);
    }
    return index;
  }
};
