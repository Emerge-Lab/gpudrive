#pragma once

#include <cstdint>

namespace madrona_gpudrive {

// Simple PRNG intended to be initialized with episode ID.
class RNG {
public:
    static inline RNG make(uint32_t idx)
    {
        uint32_t v0 = idx;
        uint32_t v1 = 0;
        uint32_t s0 = 0;

#pragma unroll
        for (int n = 0; n < 8; n++) {
            s0 += 0x9e3779b9;
            v0 += ((v1 << 4) + 0xa341316c) ^
                (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^
                ((v0 >> 5) + 0x7e95761e);
        }

        RNG rng;
        rng.v_ = v0;

        return rng;
    }
    
    inline float rand()
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        v_ = (LCG_A * v_ + LCG_C);

        uint32_t next = v_ & 0x00FFFFFF;
        return float(next) / float(0x01000000);
    }

private:
    uint32_t v_;
};

}
