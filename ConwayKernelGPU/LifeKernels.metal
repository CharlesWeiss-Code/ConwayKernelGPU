#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint {
    NMAX = 16,
    TGX  = 32,
    TGY  = 8
};

struct SimParams {
    uint   W;
    uint   H;
    uint   strideWords;
    uint   N;
    uint   subtractCenter;
    ushort kernelRow[NMAX];
    uint   _pad;
};

struct RuleLUT {
    uchar next[2][257];
};

struct ViewParams {
    uint W;
    uint H;
    uint strideWords;

    uint2  viewSizePx;
    float2 centerCell;
    float  zoom;
    uint   _pad0;

    uint aliveARGB;
    uint deadARGB;
};

inline uint wrapPow2(int v, uint mask) { return ((uint)v) & mask; }

inline ulong loadWindow64_row_pow2(
    device const uint* cur,
    uint strideWords, uint wordMask,
    uint H, uint yMask,
    int cellXStart, int y
) {
    uint wy = wrapPow2(y, yMask);

    int  word0     = cellXStart >> 5;
    uint bitOffset = ((uint)cellXStart) & 31u;

    uint w0 = ((uint)(word0 + 0)) & wordMask;
    uint w1 = ((uint)(word0 + 1)) & wordMask;
    uint w2 = ((uint)(word0 + 2)) & wordMask;

    device const uint* row = cur + wy * strideWords;

    uint a = row[w0];
    uint b = row[w1];
    uint c = row[w2];

    ulong ab = (ulong)a | ((ulong)b << 32);

    if (bitOffset == 0) {
        return ab;
    } else {
        ulong lo = (ab >> bitOffset);
        ulong hi = ((ulong)c) << (64u - bitOffset);
        return lo | hi;
    }
}

kernel void sim_step_custom_packed_pow2(
    device const uint*  cur [[buffer(0)]],
    device uint*        nxt [[buffer(1)]],
    constant SimParams& p   [[buffer(2)]],
    constant RuleLUT&   lut [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]]
) {
    uint yMask    = p.H - 1u;
    uint wordMask = p.strideWords - 1u;

    uint N = min(p.N, (uint)NMAX);
    uint R = N / 2;

    uint  tileW    = TGX + 2u * R;
    ulong tileMask = (1ull << tileW) - 1ull;

    uint xWord = tgid.x;
    uint y0    = tgid.y * TGY;

    int baseCellX  = (int)(xWord * 32u);
    int tileStartX = baseCellX - (int)R;

    threadgroup ulong tileRows[32];

    uint linear    = tid.y * TGX + tid.x;
    uint threads   = TGX * TGY;
    uint totalRows = TGY + 2u * R;

    for (uint rIdx = linear; rIdx < totalRows; rIdx += threads) {
        int y = (int)y0 + (int)rIdx - (int)R;
        ulong bits = loadWindow64_row_pow2(cur, p.strideWords, wordMask, p.H, yMask, tileStartX, y);
        tileRows[rIdx] = bits & tileMask;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint y  = y0 + tid.y;
    uint wy = ((uint)y) & yMask;

    uint windowMask = (N == 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);

    uint count = 0;
    for (uint j = 0; j < N; ++j) {
        ulong rowBits = tileRows[tid.y + j];
        uint bitsN    = (uint)((rowBits >> tid.x) & (ulong)windowMask);
        uint masked   = bitsN & (uint)(p.kernelRow[j] & (ushort)windowMask);
        count += popcount(masked);
    }

    uint centerBit = (uint)((tileRows[tid.y + R] >> (tid.x + R)) & 1ull);
    if (p.subtractCenter != 0) count -= centerBit;

    uchar alive    = (uchar)centerBit;
    uchar outAlive = lut.next[alive ? 1 : 0][min(count, 256u)];

    auto vote = simd_ballot(outAlive != 0);
    ulong mask = (ulong)vote;
    uint ballot32 = (uint)(mask & 0xFFFFFFFFul);

    if (tid.x == 0) {
        nxt[wy * p.strideWords + xWord] = ballot32;
    }
}

kernel void decode_viewport_packed_pow2(
    device const uint*   cur [[buffer(0)]],
    constant ViewParams& v   [[buffer(1)]],
    texture2d<float, access::write> outTex [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= v.viewSizePx.x || gid.y >= v.viewSizePx.y) return;

    uint wMask = v.W - 1u;
    uint yMask = v.H - 1u;

    float2 pxy = float2(gid);
    float2 viewCenterPx = 0.5f * float2(v.viewSizePx);

    float2 offsetPx = pxy - viewCenterPx;
    float2 offsetCells = offsetPx / v.zoom;
    float2 world = v.centerCell + offsetCells;

    int cx = (int)floor(world.x);
    int cy = (int)floor(world.y);

    uint wx = ((uint)cx) & wMask;
    uint wy = ((uint)cy) & yMask;

    uint word = cur[wy * v.strideWords + (wx >> 5)];
    uint bit  = (word >> (wx & 31u)) & 1u;

    float vout = (bit != 0) ? 1.0f : 0.0f;
    
    const float borderWidth = 2.0f;
    
    float normX = fmod(world.x + float(v.W) * 1000.0f, float(v.W));
    float normY = fmod(world.y + float(v.H) * 1000.0f, float(v.H));
    
    float distToX0 = min(normX, float(v.W) - normX);
    float distToY0 = min(normY, float(v.H) - normY);
    
    float pixelDistX = distToX0 * v.zoom;
    float pixelDistY = distToY0 * v.zoom;
    
    if (pixelDistX < borderWidth || pixelDistY < borderWidth) {
        outTex.write(float4(0.0f, 0.5f, 1.0f, 1.0f), gid);
        return;
    }
    
    outTex.write(float4(vout, vout, vout, 1.0f), gid);
}

kernel void draw_minimap(
    texture2d<float, access::read_write> tex [[texture(0)]],
    constant ViewParams& v [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(tex.get_width(), tex.get_height());
    
    const uint minimapSize = 300;
    const uint margin = 20;
    const uint borderWidth = 2;
    
    uint2 minimapOrigin = uint2(margin, margin);
    
    if (gid.x < minimapOrigin.x || gid.x >= minimapOrigin.x + minimapSize ||
        gid.y < minimapOrigin.y || gid.y >= minimapOrigin.y + minimapSize) {
        return;
    }
    
    uint2 localPos = gid - minimapOrigin;
    
    bool isOuterBorder = localPos.x < borderWidth ||
                         localPos.x >= minimapSize - borderWidth ||
                         localPos.y < borderWidth ||
                         localPos.y >= minimapSize - borderWidth;
    
    if (isOuterBorder) {
        tex.write(float4(1.0f, 1.0f, 1.0f, 1.0f), gid);
        return;
    }
    
    float innerSize = float(minimapSize - 2 * borderWidth);
    float2 adjustedLocal = float2(localPos) - float(borderWidth);
    
    float gridAspect = float(v.W) / float(v.H);
    float2 gridPos;
    
    if (gridAspect > 1.0f) {
        gridPos.x = (adjustedLocal.x / innerSize) * float(v.W);
        float usedHeight = innerSize / gridAspect;
        float yOffset = (innerSize - usedHeight) * 0.5f;
        gridPos.y = ((adjustedLocal.y - yOffset) / usedHeight) * float(v.H);
    } else {
        float usedWidth = innerSize * gridAspect;
        float xOffset = (innerSize - usedWidth) * 0.5f;
        gridPos.x = ((adjustedLocal.x - xOffset) / usedWidth) * float(v.W);
        gridPos.y = (adjustedLocal.y / innerSize) * float(v.H);
    }
    
    float2 viewportCells = float2(v.viewSizePx) / v.zoom;
    
    float2 viewMin = v.centerCell - viewportCells * 0.5f;
    float2 viewMax = v.centerCell + viewportCells * 0.5f;
    
    float minX = fmod(viewMin.x + float(v.W) * 1000.0f, float(v.W));
    float maxX = fmod(viewMax.x + float(v.W) * 1000.0f, float(v.W));
    float minY = fmod(viewMin.y + float(v.H) * 1000.0f, float(v.H));
    float maxY = fmod(viewMax.y + float(v.H) * 1000.0f, float(v.H));
    
    bool insideX = false;
    bool insideY = false;
    
    if (minX <= maxX) {
        insideX = (gridPos.x >= minX && gridPos.x <= maxX);
    } else {
        insideX = (gridPos.x >= minX || gridPos.x <= maxX);
    }
    
    if (minY <= maxY) {
        insideY = (gridPos.y >= minY && gridPos.y <= maxY);
    } else {
        insideY = (gridPos.y >= minY || gridPos.y <= maxY);
    }
    
    float distToMinX = abs(gridPos.x - minX);
    float distToMaxX = abs(gridPos.x - maxX);
    float distToMinY = abs(gridPos.y - minY);
    float distToMaxY = abs(gridPos.y - maxY);
    
    distToMinX = min(distToMinX, float(v.W) - distToMinX);
    distToMaxX = min(distToMaxX, float(v.W) - distToMaxX);
    distToMinY = min(distToMinY, float(v.H) - distToMinY);
    distToMaxY = min(distToMaxY, float(v.H) - distToMaxY);
    
    float distToEdgeX = min(distToMinX, distToMaxX);
    float distToEdgeY = min(distToMinY, distToMaxY);
    
    float scaleX = innerSize / float(v.W);
    float scaleY = innerSize / float(v.H);
    
    if (gridAspect > 1.0f) {
        scaleY = (innerSize / gridAspect) / float(v.H);
    } else {
        scaleX = (innerSize * gridAspect) / float(v.W);
    }
    
    float pixelDistX = distToEdgeX * scaleX;
    float pixelDistY = distToEdgeY * scaleY;
    
    bool isGreenBorder = (insideX && pixelDistY < float(borderWidth)) ||
                         (insideY && pixelDistX < float(borderWidth));
    
    if (isGreenBorder) {
        tex.write(float4(0.0f, 1.0f, 0.0f, 1.0f), gid);
    }
}

kernel void draw_kernel_rule_viz(
    texture2d<float, access::read_write> tex [[texture(0)]],
    constant SimParams& sim [[buffer(0)]],
    constant RuleLUT& lut [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 texSize = uint2(tex.get_width(), tex.get_height());
    
    const uint vizWidth = 600;
    const uint vizHeight = 450;
    const uint margin = 20;
    const uint borderWidth = 3;
    const uint padding = 20;
    
    uint2 vizOrigin = uint2(texSize.x - vizWidth - margin, texSize.y - vizHeight - margin);
    
    if (texSize.x == 0 || texSize.y == 0) return;
    
    if (gid.x < vizOrigin.x || gid.x >= vizOrigin.x + vizWidth ||
        gid.y < vizOrigin.y || gid.y >= vizOrigin.y + vizHeight) {
        return;
    }
    
    uint2 localPos = gid - vizOrigin;
    
    // Semi-transparent background
    float4 currentColor = tex.read(gid);
    float4 bgColor = float4(0.05f, 0.05f, 0.1f, 0.9f);
    float4 blended = mix(currentColor, bgColor, bgColor.a);
    
    // Outer border (cyan)
    bool isOuterBorder = localPos.x < borderWidth ||
                         localPos.x >= vizWidth - borderWidth ||
                         localPos.y < borderWidth ||
                         localPos.y >= vizHeight - borderWidth;
    
    if (isOuterBorder) {
        tex.write(float4(0.0f, 0.8f, 1.0f, 1.0f), gid);
        return;
    }
    
    tex.write(blended, gid);
    
    // ===== KERNEL VISUALIZATION =====
    const uint kernelStartY = padding + 35;
    const uint cellSize = 25;
    const uint kernelN = min(sim.N, (uint)NMAX);
    const uint kernelPixelSize = kernelN * cellSize;
    const uint kernelStartX = padding + 20;
    
    if (localPos.y >= kernelStartY && localPos.y < kernelStartY + kernelPixelSize &&
        localPos.x >= kernelStartX && localPos.x < kernelStartX + kernelPixelSize) {
        
        uint kx = (localPos.x - kernelStartX) / cellSize;
        uint ky = (localPos.y - kernelStartY) / cellSize;
        
        if (kx < kernelN && ky < kernelN) {
            bool isSet = (sim.kernelRow[ky] & (1u << kx)) != 0;
            bool isCenter = (kx == kernelN/2 && ky == kernelN/2);
            
            uint cellX = (localPos.x - kernelStartX) % cellSize;
            uint cellY = (localPos.y - kernelStartY) % cellSize;
            bool isCellBorder = (cellX == 0 || cellY == 0 || cellX == 1 || cellY == 1);
            
            if (isCellBorder) {
                tex.write(float4(0.2f, 0.2f, 0.25f, 1.0f), gid);
            } else if (isCenter && sim.subtractCenter) {
                // Center cell when subtracted (red with X)
                bool isDiagonal = (cellX == cellY) || (cellX + cellY == cellSize - 1);
                if (isDiagonal) {
                    tex.write(float4(1.0f, 0.2f, 0.2f, 1.0f), gid);
                } else if (isSet) {
                    tex.write(float4(0.7f, 0.1f, 0.1f, 1.0f), gid);
                } else {
                    tex.write(float4(0.3f, 0.05f, 0.05f, 1.0f), gid);
                }
            } else if (isSet) {
                tex.write(float4(0.3f, 1.0f, 0.4f, 1.0f), gid);
            } else {
                tex.write(float4(0.15f, 0.15f, 0.18f, 1.0f), gid);
            }
        }
    }
    
    // ===== RULE VISUALIZATION =====
    const uint ruleStartY = kernelStartY + kernelPixelSize + 40;
    const uint ruleGraphHeight = 150;
    const uint ruleGraphWidth = min(500u, vizWidth - padding * 2 - 40);
    const uint ruleStartX = padding + 20;
    
    // Calculate max possible neighbors based on kernel size
    uint maxNeighbors = kernelN * kernelN;
    if (sim.subtractCenter) maxNeighbors -= 1;
    // Add some padding to the scale
    maxNeighbors = min(maxNeighbors + 2, 256u);
    
    if (localPos.y >= ruleStartY && localPos.y < ruleStartY + ruleGraphHeight &&
        localPos.x >= ruleStartX && localPos.x < ruleStartX + ruleGraphWidth) {
        
        uint xPos = localPos.x - ruleStartX;
        uint yPos = localPos.y - ruleStartY;
        
        // Map x position to neighbor count (0 to maxNeighbors)
        uint count = (xPos * (maxNeighbors + 1)) / ruleGraphWidth;
        if (count > maxNeighbors) count = maxNeighbors;
        
        const uint halfHeight = ruleGraphHeight / 2;
        const uint dividerY = halfHeight;
        const uint dividerThickness = 3;
        
        // Divider line (thicker)
        if (yPos >= dividerY - dividerThickness/2 && yPos <= dividerY + dividerThickness/2) {
            tex.write(float4(0.5f, 0.5f, 0.55f, 1.0f), gid);
            return;
        }
        
        bool isBirthSection = (yPos < dividerY);
        
        // Each count gets multiple pixels width
        const uint pixelsPerCount = max(10u, ruleGraphWidth / (maxNeighbors + 1));
        
        if (isBirthSection) {
            // Birth rules (top half)
            bool isActive = (lut.next[0][count] != 0);
            
            uint barHeight = halfHeight - 8;
            uint barY = dividerY - yPos - dividerThickness/2;
            
            if (barY <= barHeight && isActive) {
                float intensity = 1.0f - (float(barY) / float(barHeight)) * 0.2f;
                tex.write(float4(1.0f * intensity, 0.9f * intensity, 0.2f * intensity, 1.0f), gid);
            }
        } else {
            // Survive rules (bottom half)
            bool isActive = (lut.next[1][count] != 0);
            
            uint barHeight = halfHeight - 8;
            uint barY = yPos - dividerY - dividerThickness/2 - 1;
            
            if (barY <= barHeight && isActive) {
                float intensity = 1.0f - (float(barY) / float(barHeight)) * 0.2f;
                tex.write(float4(0.2f * intensity, 0.9f * intensity, 1.0f * intensity, 1.0f), gid);
            }
        }
    }
    
    // Add tick marks on x-axis to show neighbor count scale
    const uint tickY = ruleStartY + ruleGraphHeight + 5;
    if (localPos.y >= tickY && localPos.y < tickY + 15 &&
        localPos.x >= ruleStartX && localPos.x < ruleStartX + ruleGraphWidth) {
        
        uint xPos = localPos.x - ruleStartX;
        
        // Draw tick marks for each neighbor count (0 to maxNeighbors)
        for (uint tickVal = 0; tickVal <= maxNeighbors; ++tickVal) {
            uint tickX = (tickVal * ruleGraphWidth) / (maxNeighbors + 1) + (ruleGraphWidth / (maxNeighbors + 1)) / 2;
            
            // Draw vertical tick line
            if (xPos >= tickX - 1 && xPos <= tickX + 1) {
                if (localPos.y < tickY + 8) {
                    tex.write(float4(0.7f, 0.7f, 0.75f, 1.0f), gid);
                }
                return;
            }
        }
    }
    
    // Title bar at top
    if (localPos.y >= padding - 3 && localPos.y < padding + 12 &&
        localPos.x >= padding + 10 && localPos.x < vizWidth - padding - 10) {
        tex.write(float4(0.9f, 0.9f, 0.95f, 1.0f), gid);
    }
}
