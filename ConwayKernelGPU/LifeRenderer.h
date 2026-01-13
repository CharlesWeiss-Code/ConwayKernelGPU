#pragma once
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

@interface LifeRenderer : NSObject
- (instancetype)initWithLayer:(CAMetalLayer*)layer;

- (void)resizeDrawableIfNeeded;
- (void)drawFrame;

// Set from view/window when backing scale changes (Retina)
- (void)setBackingScale:(float)scale;

// Camera control (in drawable pixel space)
- (void)panByDeltaPixelsX:(float)dx y:(float)dy;
- (void)zoomAtPixelX:(float)mx y:(float)my wheelDelta:(float)delta;
- (void)zoomToFit;
- (void)resetView;
- (void)setZoom:(float)z;

// Kernel config
- (void)setKernelHex:(NSString*)hexString subtractCenter:(BOOL)subtractCenter N:(uint32_t)N;

// Rule config
- (void)setClassicLifeRule;
- (void)setRuleBirthRange:(int)b0 b1:(int)b1 surviveRange:(int)s0 s1:(int)s1;
- (BOOL)setRuleString:(NSString*)ruleString;

// Seeding
- (void)seedRandomSparse:(float)density;
- (void)seedCheckerboard;
- (void)seedRandomDense;
- (void)seedGliderGun;

- (void)resetSimulation;


// Pause control
- (void)togglePause;
- (BOOL)isPaused;

- (void)toggleMinimap;
- (BOOL)isMinimapVisible;

- (void)toggleVisualizer;
- (BOOL)isVisualizerVisible;

@end
