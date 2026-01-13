#import "LifeRenderer.h"
#import "LifeShared.h"

#import <simd/simd.h>
#import <Metal/Metal.h>

#include "KernelHex.h"
#include "LifeRules.h"
#include <array>
#include <random>
#include <string>

static const NSUInteger kTGY = 8;

static inline uint32_t NextPow2(uint32_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

@implementation LifeRenderer {
    CAMetalLayer* _layer;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;

    id<MTLComputePipelineState> _psoSim;
    id<MTLComputePipelineState> _psoDecode;
    id<MTLComputePipelineState> _psoMinimap;
    id<MTLComputePipelineState> _psoVisualizer;


    id<MTLBuffer> _bufA;
    id<MTLBuffer> _bufB;

    id<MTLBuffer> _simParamsBuf;
    id<MTLBuffer> _lutBuf;
    id<MTLBuffer> _viewParamsBuf;

    uint32_t _W, _H, _strideWords;

    simd_float2 _centerCell;
    float _zoom;

    id<MTLBuffer> _cur;
    id<MTLBuffer> _nxt;

    float _backingScale;

    dispatch_semaphore_t _inFlight;
    BOOL _paused;
    BOOL _showMinimap;
    BOOL _showVisualizer;

}

#pragma mark - Init

- (instancetype)initWithLayer:(CAMetalLayer*)layer {
    if (!(self = [super init])) return nil;
    _layer = layer;

    _device = MTLCreateSystemDefaultDevice();
    _queue  = [_device newCommandQueue];

    _layer.device = _device;
    _layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    _layer.framebufferOnly = NO;
    _layer.presentsWithTransaction = NO;

    if ([_layer respondsToSelector:@selector(setMaximumDrawableCount:)]) {
        _layer.maximumDrawableCount = 3;
    }
    if ([_layer respondsToSelector:@selector(setAllowsNextDrawableTimeout:)]) {
        _layer.allowsNextDrawableTimeout = YES;
    }
    _inFlight = dispatch_semaphore_create(3);

    _backingScale = 1.0f;
    _zoom = 16.0f;
    _paused = NO;

    [self _createSimulationWithW:512 H:512];
    [self _createPipelines];

    [self setKernelHex:@"0007000700070000000000000000000000000000000000000000000000000000"
        subtractCenter:YES N:3];
    
    [self setClassicLifeRule];
//    [self seedRandomDense];
//    [self seedCheckerboard];
    [self seedGliderGun];
    [self zoomToFit];

    return self;
}

- (void)_createSimulationWithW:(uint32_t)W H:(uint32_t)H {
    W = NextPow2(W);
    H = NextPow2(H);
    if (W < 32) W = 32;
    W = (W + 31u) & ~31u;

    _W = W;
    _H = H;
    _strideWords = _W >> 5;

    size_t wordCount = (size_t)_strideWords * (size_t)_H;
    size_t byteCount = wordCount * sizeof(uint32_t);

    _bufA = [_device newBufferWithLength:byteCount options:MTLResourceStorageModePrivate];
    _bufB = [_device newBufferWithLength:byteCount options:MTLResourceStorageModePrivate];

    _cur = _bufA;
    _nxt = _bufB;

    _centerCell = (simd_float2){ (float)_W * 0.5f, (float)_H * 0.5f };
}

- (void)_createPipelines {
    NSError* err = nil;

    id<MTLLibrary> lib = [_device newDefaultLibrary];
    if (!lib) {
        NSLog(@"ERROR: newDefaultLibrary failed.");
        return;
    }

    id<MTLFunction> fSim = [lib newFunctionWithName:@"sim_step_custom_packed_pow2"];
    id<MTLFunction> fDec = [lib newFunctionWithName:@"decode_viewport_packed_pow2"];
    id<MTLFunction> fMinimap = [lib newFunctionWithName:@"draw_minimap"];
    id<MTLFunction> fVisualizer = [lib newFunctionWithName:@"draw_kernel_rule_viz"];

    _psoSim = [_device newComputePipelineStateWithFunction:fSim error:&err];
    if (!_psoSim) NSLog(@"PSO sim error: %@", err);

    _psoDecode = [_device newComputePipelineStateWithFunction:fDec error:&err];
    if (!_psoDecode) NSLog(@"PSO decode error: %@", err);

    _psoMinimap = [_device newComputePipelineStateWithFunction:fMinimap error:&err];
    if (!_psoMinimap) NSLog(@"PSO minimap error: %@", err);

    _psoVisualizer = [_device newComputePipelineStateWithFunction:fVisualizer error:&err];
    if (!_psoVisualizer) {
        NSLog(@"PSO visualizer error: %@", err);
    } else {
        NSLog(@"✅ Visualizer pipeline created successfully");
    }

    _simParamsBuf  = [_device newBufferWithLength:sizeof(SimParams) options:MTLResourceStorageModeShared];
    _lutBuf        = [_device newBufferWithLength:sizeof(RuleLUT)   options:MTLResourceStorageModeShared];
    _viewParamsBuf = [_device newBufferWithLength:sizeof(ViewParams) options:MTLResourceStorageModeShared];

    SimParams* sp = (SimParams*)_simParamsBuf.contents;
    sp->W = _W;
    sp->H = _H;
    sp->strideWords = _strideWords;
    sp->N = 16;
    sp->subtractCenter = 1;
    sp->_pad = 0;
    for (int i=0;i<16;i++) sp->kernelRow[i] = 0;

    ViewParams* vp = (ViewParams*)_viewParamsBuf.contents;
    vp->W = _W;
    vp->H = _H;
    vp->strideWords = _strideWords;
    vp->centerCell = _centerCell;
    vp->zoom = _zoom;
    vp->aliveARGB = 0xFFFFFFFF;
    vp->deadARGB  = 0xFF000000;

    RuleLUT* rl = (RuleLUT*)_lutBuf.contents;
    BuildRuleLUT_Ranges(rl->next, 36, 42, 30, 55);
    
    _showMinimap = YES;
    _showVisualizer = YES;
}

#pragma mark - Rules

- (void)setClassicLifeRule {
    RuleLUT* rl = (RuleLUT*)_lutBuf.contents;
    BuildRuleLUT_ClassicLife(rl->next);
}

- (void)setRuleBirthRange:(int)b0 b1:(int)b1 surviveRange:(int)s0 s1:(int)s1 {
    RuleLUT* rl = (RuleLUT*)_lutBuf.contents;
    BuildRuleLUT_Ranges(rl->next, b0, b1, s0, s1);
}

- (BOOL)setRuleString:(NSString*)ruleString {
    RuleLUT* rl = (RuleLUT*)_lutBuf.contents;
    std::string s([ruleString UTF8String]);
    return BuildRuleLUT_FromString(rl->next, s.c_str());
}

#pragma mark - Kernel

- (void)setKernelHex:(NSString*)hexString subtractCenter:(BOOL)subtractCenter N:(uint32_t)N {
    std::string s([[hexString stringByTrimmingCharactersInSet:
                    NSCharacterSet.whitespaceAndNewlineCharacterSet] UTF8String]);

    std::array<uint16_t,16> rows;
    if (!ParseKernel16Hex(s, rows)) {
        NSLog(@"Kernel hex parse failed.");
        return;
    }

    if (N == 0) N = 1;
    if (N > 16) N = 16;

    SimParams* sp = (SimParams*)_simParamsBuf.contents;
    sp->W = _W;
    sp->H = _H;
    sp->strideWords = _strideWords;
    sp->N = N;
    sp->subtractCenter = subtractCenter ? 1u : 0u;

    for (int i=0;i<16;i++) sp->kernelRow[i] = 0;
    for (uint32_t i=0; i<N; ++i) sp->kernelRow[i] = rows[i];
}

#pragma mark - Seeding

- (void)seedRandomSparse:(float)density {
    size_t wordCount = (size_t)_strideWords * (size_t)_H;
    size_t byteCount = wordCount * sizeof(uint32_t);

    id<MTLBuffer> staging = [_device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
    uint32_t* dst = (uint32_t*)staging.contents;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    for (size_t i = 0; i < wordCount; ++i) {
        uint32_t w = 0;
        for (int b = 0; b < 32; ++b) {
            if (prob(rng) < density) w |= (1u << b);
        }
        dst[i] = w;
    }

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:_cur destinationOffset:0 size:byteCount];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

- (void)seedCheckerboard {
    size_t wordCount = (size_t)_strideWords * (size_t)_H;
    size_t byteCount = wordCount * sizeof(uint32_t);

    id<MTLBuffer> staging = [_device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
    uint32_t* dst = (uint32_t*)staging.contents;

    for (uint32_t y = 0; y < _H; ++y) {
        uint32_t rowPat = (y & 1) ? 0xAAAAAAAAu : 0x55555555u;
        for (uint32_t xw = 0; xw < _strideWords; ++xw) {
            dst[(size_t)y * _strideWords + xw] = rowPat;
        }
    }

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:_cur destinationOffset:0 size:byteCount];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

- (void)seedGliderGun {
    size_t wordCount = (size_t)_strideWords * (size_t)_H;
    size_t byteCount = wordCount * sizeof(uint32_t);

    id<MTLBuffer> staging = [_device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
    uint32_t* dst = (uint32_t*)staging.contents;
    
    memset(dst, 0, byteCount);
    
    auto setBit = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= (int)_W || y >= (int)_H) return;
        uint32_t word = x / 32;
        uint32_t bit = x % 32;
        dst[y * _strideWords + word] |= (1u << bit);
    };
    
    for (int gy = 200; gy < (int)_H - 200; gy += 400) {
        for (int gx = 200; gx < (int)_W - 200; gx += 400) {
            int gun[][2] = {
                {24,0}, {22,1}, {24,1}, {12,2}, {13,2}, {20,2}, {21,2}, {34,2}, {35,2},
                {11,3}, {15,3}, {20,3}, {21,3}, {34,3}, {35,3}, {0,4}, {1,4}, {10,4},
                {16,4}, {20,4}, {21,4}, {0,5}, {1,5}, {10,5}, {14,5}, {16,5}, {17,5},
                {22,5}, {24,5}, {10,6}, {16,6}, {24,6}, {11,7}, {15,7}, {12,8}, {13,8}
            };
            
            for (auto& p : gun) {
                setBit(gx + p[0], gy + p[1]);
            }
        }
    }

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:_cur destinationOffset:0 size:byteCount];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    NSLog(@"=== GLIDER GUN SEEDED ===");
}

- (void)seedRandomDense {
    size_t wordCount = (size_t)_strideWords * (size_t)_H;
    size_t byteCount = wordCount * sizeof(uint32_t);

    id<MTLBuffer> staging = [_device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
    uint32_t* dst = (uint32_t*)staging.contents;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    float density = 0.375f;
    
    for (size_t i = 0; i < wordCount; ++i) {
        uint32_t w = 0;
        for (int b = 0; b < 32; ++b) {
            if (prob(rng) < density) w |= (1u << b);
        }
        dst[i] = w;
    }

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:_cur destinationOffset:0 size:byteCount];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    NSLog(@"=== RANDOM DENSE SEEDED (%.1f%%) ===", density * 100);
}

#pragma mark - Drawable / Camera

- (void)resizeDrawableIfNeeded {
    if (!_layer) return;
    CGSize logical = _layer.bounds.size;
    CGSize drawable = CGSizeMake(logical.width * _backingScale, logical.height * _backingScale);
    if (!CGSizeEqualToSize(_layer.drawableSize, drawable)) {
        _layer.drawableSize = drawable;
    }
}

- (void)setBackingScale:(float)scale {
    _backingScale = (scale > 0.0f) ? scale : 1.0f;
    [self resizeDrawableIfNeeded];
}

- (void)panByDeltaPixelsX:(float)dx y:(float)dy {
    _centerCell -= (simd_float2){ dx / _zoom, dy / _zoom };
}

- (void)zoomAtPixelX:(float)mx y:(float)my wheelDelta:(float)delta {
    float factor = powf(1.1f, -delta / 50.0f);
    
    CGSize ds = _layer.drawableSize;
    float minZoomX = ds.width / (float)_W;
    float minZoomY = ds.height / (float)_H;
    float minZoom = fminf(minZoomX, minZoomY) * 0.9f;
    
    float newZoom = fminf(fmaxf(_zoom * factor, minZoom), 256.0f);

    simd_float2 viewSizePx = { (float)ds.width, (float)ds.height };
    simd_float2 viewCenterPx = viewSizePx * 0.5f;
    simd_float2 mousePx = { mx, my };

    simd_float2 before = _centerCell + (mousePx - viewCenterPx) / _zoom;
    _zoom = newZoom;
    simd_float2 after  = _centerCell + (mousePx - viewCenterPx) / _zoom;
    _centerCell += (before - after);
}

- (void)zoomToFit {
    CGSize ds = _layer.drawableSize;
    
    if (ds.width <= 0 || ds.height <= 0) {
        NSLog(@"⚠️ Invalid drawable size for zoom calculation");
        return;
    }
    
    float zoomX = ds.width / (float)_W;
    float zoomY = ds.height / (float)_H;
    
    _zoom = fminf(zoomX, zoomY) * 0.9f;
    _zoom = fmaxf(0.1f, fminf(_zoom, 256.0f));
    
    _centerCell = (simd_float2){ (float)_W * 0.5f, (float)_H * 0.5f };
    
    NSLog(@"Zoom to fit: grid=%ux%u drawable=%.0fx%.0f final=%.2f",
          _W, _H, ds.width, ds.height, _zoom);
}

- (void)resetView {
    [self zoomToFit];
}

- (void)setZoom:(float)z {
    _zoom = fmaxf(0.1f, fminf(z, 256.0f));
}

#pragma mark - Pause Control

- (void)togglePause {
    _paused = !_paused;
    NSLog(@"Simulation %@", _paused ? @"PAUSED" : @"RESUMED");
}

- (BOOL)isPaused {
    return _paused;
}

#pragma mark - Frame

- (void)drawFrame {
    @autoreleasepool {
        if (!_psoSim || !_psoDecode) return;

        id<CAMetalDrawable> drawable = [_layer nextDrawable];
        if (!drawable) return;

        ViewParams* vp = (ViewParams*)_viewParamsBuf.contents;
        vp->W = _W;
        vp->H = _H;
        vp->strideWords = _strideWords;
        vp->viewSizePx = (simd_uint2){ (uint32_t)drawable.texture.width, (uint32_t)drawable.texture.height };
        vp->centerCell = _centerCell;
        vp->zoom = _zoom;

        SimParams* sp = (SimParams*)_simParamsBuf.contents;
        sp->W = _W;
        sp->H = _H;
        sp->strideWords = _strideWords;

        id<MTLCommandBuffer> cb = [_queue commandBuffer];

        // Only run simulation if not paused
        if (!_paused) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:_psoSim];
            [enc setBuffer:_cur offset:0 atIndex:0];
            [enc setBuffer:_nxt offset:0 atIndex:1];
            [enc setBuffer:_simParamsBuf offset:0 atIndex:2];
            [enc setBuffer:_lutBuf offset:0 atIndex:3];

            const uint32_t tgY = (sp->H + (uint32_t)kTGY - 1u) / (uint32_t)kTGY;
            MTLSize tgs = MTLSizeMake(sp->strideWords, tgY, 1);
            MTLSize tpt = MTLSizeMake(32, kTGY, 1);

            [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpt];
            [enc endEncoding];
            
            // Swap buffers
            id<MTLBuffer> tmp = _cur; _cur = _nxt; _nxt = tmp;
        }

        // DECODE (always render, even when paused)
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:_psoDecode];
            [enc setBuffer:_cur offset:0 atIndex:0];
            [enc setBuffer:_viewParamsBuf offset:0 atIndex:1];
            [enc setTexture:drawable.texture atIndex:0];

            MTLSize threads = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake(drawable.texture.width, drawable.texture.height, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:threads];
            [enc endEncoding];
        }

        // MINIMAP (draw on top of decoded image)
        if (_showMinimap && _psoMinimap) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:_psoMinimap];
            [enc setTexture:drawable.texture atIndex:0];
            [enc setBuffer:_viewParamsBuf offset:0 atIndex:0];

            MTLSize threads = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake(drawable.texture.width, drawable.texture.height, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:threads];
            [enc endEncoding];
        }

        // VISUALIZER (draw kernel and rule info)
        if (_showVisualizer && _psoVisualizer) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:_psoVisualizer];
            [enc setTexture:drawable.texture atIndex:0];
            [enc setBuffer:_simParamsBuf offset:0 atIndex:0];
            [enc setBuffer:_lutBuf offset:0 atIndex:1];

            MTLSize threads = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake(drawable.texture.width, drawable.texture.height, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:threads];
            [enc endEncoding];
            
        }

        [cb presentDrawable:drawable];
        [cb commit];
    }
}

#pragma mark - Minimap Control

- (void)toggleMinimap {
    _showMinimap = !_showMinimap;
    NSLog(@"Minimap %@", _showMinimap ? @"SHOWN" : @"HIDDEN");
}

- (BOOL)isMinimapVisible {
    return _showMinimap;
}

#pragma mark - Visualizer Control

- (void)toggleVisualizer {
    _showVisualizer = !_showVisualizer;
    NSLog(@"Visualizer %@ (PSO exists: %d)", _showVisualizer ? @"SHOWN" : @"HIDDEN", _psoVisualizer != nil);
}

- (BOOL)isVisualizerVisible {
    return _showVisualizer;
}

#pragma mark - Reset

- (void)resetSimulation {
    // Clear current state
    size_t wordCount = (size_t)_strideWords * (size_t)_H;
    size_t byteCount = wordCount * sizeof(uint32_t);
    
    id<MTLBuffer> staging = [_device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
    uint32_t* dst = (uint32_t*)staging.contents;
    
    // Generate new random seed with different RNG seed
    static uint32_t resetCount = 0;
    resetCount++;
    
    std::mt19937 rng((uint32_t)time(nullptr) + resetCount);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    
    float density = 0.375f;
    
    for (size_t i = 0; i < wordCount; ++i) {
        uint32_t w = 0;
        for (int b = 0; b < 32; ++b) {
            if (prob(rng) < density) w |= (1u << b);
        }
        dst[i] = w;
    }
    
    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:_cur destinationOffset:0 size:byteCount];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    NSLog(@"=== SIMULATION RESET (seed #%u) ===", resetCount);
}

@end
