#pragma once
#include <array>
#include <cstdint>

// lut[alive][count] -> 0/1
// alive = 0 (dead), 1 (alive)
// count = 0..256

void BuildRuleLUT_BS(uint8_t lut[2][257],
                     const std::array<bool,257>& born,
                     const std::array<bool,257>& survive);

// Classic Conway: B3/S23
void BuildRuleLUT_ClassicLife(uint8_t lut[2][257]);

// Inclusive ranges: birth in [b0,b1], survive in [s0,s1]
void BuildRuleLUT_Ranges(uint8_t lut[2][257],
                         int b0, int b1,
                         int s0, int s1);

// Convenience: single-value birth/survive sets (inclusive)
void BuildRuleLUT_Singles(uint8_t lut[2][257],
                          int b, int s0, int s1);

// Parse "B.../S..." where numbers can be multi-digit (0..256),
// separated by commas or spaces. Example: "B3/S2,3" or "B36-42/S30-55".
bool BuildRuleLUT_FromString(uint8_t lut[2][257], const char* rule);
