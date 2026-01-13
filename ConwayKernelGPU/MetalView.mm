// MetalView.mm
#import "MetalView.h"
#import <QuartzCore/CAMetalLayer.h>
#import <CoreVideo/CoreVideo.h>
#import "LifeRenderer.h"

@implementation MetalView {
    CAMetalLayer* _metalLayer;
    LifeRenderer* _renderer;

    CVDisplayLinkRef _displayLink;

    BOOL _frameScheduled;

    BOOL _dragging;
    NSPoint _lastMouseLogical;
}

// ✅ macOS: NSView must provide a CAMetalLayer backing layer this way
- (CALayer *)makeBackingLayer {
    return [CAMetalLayer layer];
}

- (instancetype)initWithFrame:(NSRect)frameRect {
    if (!(self = [super initWithFrame:frameRect])) return nil;

    self.wantsLayer = YES;
    _metalLayer = (CAMetalLayer*)self.layer; // now really a CAMetalLayer

    // Initial sizing (prevents drawableSize=0 early)
    CGFloat scale = NSScreen.mainScreen.backingScaleFactor ?: 1.0;
    _metalLayer.contentsScale = scale;
    _metalLayer.frame = self.bounds;
    _metalLayer.drawableSize = CGSizeMake(frameRect.size.width * scale, frameRect.size.height * scale);

    _renderer = [[LifeRenderer alloc] initWithLayer:_metalLayer];

    [self _updateBackingScale];
    [_renderer resizeDrawableIfNeeded];

    [self _setupDisplayLink];

    return self;
}

- (BOOL)acceptsFirstResponder { return YES; }

- (void)viewDidMoveToWindow {
    [super viewDidMoveToWindow];
    [self _updateBackingScale];
    [self setNeedsLayout:YES];
}

- (void)layout {
    [super layout];

    if (!_metalLayer) return;

    _metalLayer.frame = self.bounds;

    CGFloat scale = self.window.backingScaleFactor ?: NSScreen.mainScreen.backingScaleFactor ?: 1.0;
    _metalLayer.contentsScale = scale;

    CGSize logical = self.bounds.size;
    CGSize drawable = CGSizeMake(logical.width * scale, logical.height * scale);
    
    if (!CGSizeEqualToSize(_metalLayer.drawableSize, drawable)) {
        _metalLayer.drawableSize = drawable;
    }

    [_renderer setBackingScale:(float)scale];
    [_renderer resizeDrawableIfNeeded];
    
    // ✅ Recalculate zoom when window resizes
    static BOOL firstLayout = YES;
    if (firstLayout) {
        firstLayout = NO;
        [_renderer zoomToFit];
    }
}

- (void)setFrameSize:(NSSize)newSize {
    [super setFrameSize:newSize];
    [self setNeedsLayout:YES];
}

- (void)_updateBackingScale {
    CGFloat scale = self.window.backingScaleFactor ?: NSScreen.mainScreen.backingScaleFactor ?: 1.0;
    [_renderer setBackingScale:(float)scale];
}

- (void)_setupDisplayLink {
    if (_displayLink) return;

    CVDisplayLinkCreateWithActiveCGDisplays(&_displayLink);
    CVDisplayLinkSetOutputCallback(_displayLink, &DisplayLinkCallback, (__bridge void*)self);
    CVDisplayLinkStart(_displayLink);
}

static CVReturn DisplayLinkCallback(CVDisplayLinkRef displayLink,
                                   const CVTimeStamp* now,
                                   const CVTimeStamp* outputTime,
                                   CVOptionFlags flagsIn,
                                   CVOptionFlags* flagsOut,
                                   void* displayLinkContext)
{
    (void)displayLink; (void)now; (void)outputTime; (void)flagsIn; (void)flagsOut;

    @autoreleasepool {
        MetalView* view = (__bridge MetalView*)displayLinkContext;

        // ✅ Don’t queue infinite drawFrame calls (prevents drawable starvation)
        if (view->_frameScheduled) return kCVReturnSuccess;
        view->_frameScheduled = YES;

        dispatch_async(dispatch_get_main_queue(), ^{
            view->_frameScheduled = NO;
            [view drawFrame];
        });
    }
    return kCVReturnSuccess;
}

- (void)drawFrame {
    if (!self.window) return;
    [_renderer drawFrame];
}

// --- Input: Pan (drag) and Zoom (scroll) ---
// Events are in logical points; convert to drawable pixels using backing scale.

- (void)mouseDown:(NSEvent *)event {
    _dragging = YES;
    _lastMouseLogical = [self convertPoint:event.locationInWindow fromView:nil];
}

- (void)mouseUp:(NSEvent *)event {
    (void)event;
    _dragging = NO;
}

- (void)mouseDragged:(NSEvent *)event {
    if (!_dragging) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    CGFloat scale = self.window.backingScaleFactor ?: 1.0;

    float dx = (float)((p.x - _lastMouseLogical.x) * scale);
    float dy = -(float)((p.y - _lastMouseLogical.y) * scale);
    _lastMouseLogical = p;

    [_renderer panByDeltaPixelsX:dx y:dy];
}

- (void)scrollWheel:(NSEvent *)event {
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    CGFloat scale = self.window.backingScaleFactor ?: 1.0;
    
    // ✅ Get drawable size to flip Y coordinate
    CGSize drawableSize = _metalLayer.drawableSize;

    float mx = (float)(p.x * scale);
    float my = (float)(drawableSize.height - p.y * scale);  // ✅ Flip Y to match drag coordinate system
    float delta = (float)event.scrollingDeltaY;

    [_renderer zoomAtPixelX:mx y:my wheelDelta:delta];
}

- (void)keyDown:(NSEvent *)event {
    NSString* key = event.charactersIgnoringModifiers;
    
    if ([key isEqualToString:@"p"]) {
        [_renderer togglePause];
    } else if ([key isEqualToString:@"m"]) {
        [_renderer toggleMinimap];
    } else if ([key isEqualToString:@"v"]) {
        [_renderer toggleVisualizer];
    } else if ([key isEqualToString:@"r"]) {
        [_renderer resetSimulation];
    } else if ([key isEqualToString:@"f"] || [key isEqualToString:@"0"]) {
        [_renderer zoomToFit];
    } else if ([key isEqualToString:@"1"]) {
        [_renderer setZoom:1.0f];
    } else {
        [super keyDown:event];
    }
}

- (void)dealloc {
    if (_displayLink) {
        CVDisplayLinkStop(_displayLink);
        CVDisplayLinkRelease(_displayLink);
        _displayLink = NULL;
    }
}

@end
