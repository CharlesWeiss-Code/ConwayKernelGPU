#include "LifeRules.h"
#include <cstring>
#include <cctype>
#include <algorithm>

static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

void BuildRuleLUT_BS(uint8_t lut[2][257],
                     const std::array<bool,257>& born,
                     const std::array<bool,257>& survive)
{
    for (int a = 0; a < 2; ++a) {
        for (int c = 0; c <= 256; ++c) {
            bool out = (a == 0) ? born[c] : survive[c];
            lut[a][c] = out ? 1 : 0;
        }
    }
}

void BuildRuleLUT_ClassicLife(uint8_t lut[2][257]) {
    std::memset(lut, 0, sizeof(uint8_t) * 2 * 257);
    lut[0][3] = 1; // birth at 3
    lut[1][2] = 1; // survive at 2
    lut[1][3] = 1; // survive at 3
}

void BuildRuleLUT_Ranges(uint8_t lut[2][257],
                         int b0, int b1,
                         int s0, int s1)
{
    b0 = clampi(b0, 0, 256); b1 = clampi(b1, 0, 256);
    s0 = clampi(s0, 0, 256); s1 = clampi(s1, 0, 256);
    if (b0 > b1) std::swap(b0, b1);
    if (s0 > s1) std::swap(s0, s1);

    for (int c = 0; c <= 256; ++c) {
        lut[0][c] = (c >= b0 && c <= b1) ? 1 : 0;
        lut[1][c] = (c >= s0 && c <= s1) ? 1 : 0;
    }
}

void BuildRuleLUT_Singles(uint8_t lut[2][257], int b, int s0, int s1) {
    std::memset(lut, 0, sizeof(uint8_t) * 2 * 257);
    b  = clampi(b,  0, 256);
    s0 = clampi(s0, 0, 256);
    s1 = clampi(s1, 0, 256);
    if (s0 > s1) std::swap(s0, s1);

    lut[0][b] = 1;
    for (int c = s0; c <= s1; ++c) lut[1][c] = 1;
}

// --- Rule string parser ---
// Supports:
//   "B3/S23" (digits can be multi-digit; adjacency means separate numbers only if comma/space given)
//   "B3/S2,3"
//   "B36-42/S30-55"
// Digits without separators are interpreted as a single multi-digit number, e.g. "B23" -> birth=23 (not 2 and 3).
static bool parseListInto(std::array<bool,257>& set, const char* s) {
    // s points at first char after 'B' or 'S'
    // Stop on '/' or end.
    while (*s && *s != '/') {
        while (*s && *s != '/' && !std::isdigit((unsigned char)*s)) s++;

        if (!*s || *s == '/') break;

        // parse number
        int a = 0;
        while (std::isdigit((unsigned char)*s)) {
            a = a * 10 + (*s - '0');
            s++;
        }
        a = clampi(a, 0, 256);

        // optional range "-b"
        if (*s == '-') {
            s++;
            int b = 0;
            if (!std::isdigit((unsigned char)*s)) return false;
            while (std::isdigit((unsigned char)*s)) {
                b = b * 10 + (*s - '0');
                s++;
            }
            b = clampi(b, 0, 256);
            if (a > b) std::swap(a, b);
            for (int v = a; v <= b; ++v) set[v] = true;
        } else {
            set[a] = true;
        }

        // allow separators: comma/space
        while (*s && *s != '/' && (*s == ',' || std::isspace((unsigned char)*s))) s++;
    }
    return true;
}

bool BuildRuleLUT_FromString(uint8_t lut[2][257], const char* rule) {
    if (!rule) return false;

    std::array<bool,257> born{};
    std::array<bool,257> survive{};

    const char* p = rule;
    while (*p && std::isspace((unsigned char)*p)) p++;

    // Find B...
    const char* bpos = nullptr;
    const char* spos = nullptr;
    for (const char* q = p; *q; ++q) {
        if (*q == 'B' || *q == 'b') bpos = q + 1;
        if (*q == 'S' || *q == 's') spos = q + 1;
    }
    if (!bpos || !spos) return false;

    // Parse B list until '/'
    if (!parseListInto(born, bpos)) return false;
    if (!parseListInto(survive, spos)) return false;

    BuildRuleLUT_BS(lut, born, survive);
    return true;
}
