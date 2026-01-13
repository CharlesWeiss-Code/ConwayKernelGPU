#include "KernelHex.h"
#include <cctype>

static int hexVal(char c){
    if (c >= '0' && c <= '9') return c - '0';
    c = (char)std::tolower((unsigned char)c);
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    return -1;
}

bool ParseKernel16Hex(const std::string& s, std::array<uint16_t,16>& outRows) {
    std::string h;
    h.reserve(64);
    for (char c : s) {
        if (hexVal(c) >= 0) h.push_back(c);
    }
    if (h.size() != 64) return false;

    for (int r = 0; r < 16; ++r) {
        uint16_t row = 0;
        for (int k = 0; k < 4; ++k) {
            int v = hexVal(h[r*4 + k]);
            row = (uint16_t)((row << 4) | (uint16_t)v);
        }
        outRows[r] = row;
    }
    return true;
}

uint16_t ReverseBits16(uint16_t v) {
    v = (uint16_t)(((v & 0x5555u) << 1) | ((v & 0xAAAAu) >> 1));
    v = (uint16_t)(((v & 0x3333u) << 2) | ((v & 0xCCCCu) >> 2));
    v = (uint16_t)(((v & 0x0F0Fu) << 4) | ((v & 0xF0F0u) >> 4));
    v = (uint16_t)((v << 8) | (v >> 8));
    return v;
}
