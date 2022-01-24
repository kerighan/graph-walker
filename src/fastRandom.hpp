#include <stdlib.h>
#include <time.h>
#include <stdint.h>

uint64_t s[2] = {(uint64_t)time(nullptr), (uint64_t)std::rand()};
uint64_t xorshift128(void)
{
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23;
    return (s[1] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
}

float xorshift128f()
{
    return (float)xorshift128() / 18446744073709551615;
}
