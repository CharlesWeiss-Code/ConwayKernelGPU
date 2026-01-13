#pragma once
#include <array>
#include <string>
#include <cstdint>

bool ParseKernel16Hex(const std::string& s, std::array<uint16_t,16>& outRows);
uint16_t ReverseBits16(uint16_t v);
