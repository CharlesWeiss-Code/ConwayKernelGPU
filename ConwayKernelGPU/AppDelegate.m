#import "AppDelegate.h"
#import "MetalView.h"

@implementation AppDelegate {
    NSWindow* _window;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    (void)notification;
    NSRect frame = NSMakeRect(0, 0, 1200, 800);

    _window = [[NSWindow alloc] initWithContentRect:frame
                                          styleMask:(NSWindowStyleMaskTitled |
                                                     NSWindowStyleMaskClosable |
                                                     NSWindowStyleMaskResizable)
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"Metal Packed Life (Zoom/Pan)";
    _window.contentView = [[MetalView alloc] initWithFrame:frame];
    
    NSView *content = _window.contentView;
    MetalView *v = [[MetalView alloc] initWithFrame:content.bounds];
    v.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _window.contentView = v;

    
    [_window makeKeyAndOrderFront:nil];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender;
    return YES;
}

@end
