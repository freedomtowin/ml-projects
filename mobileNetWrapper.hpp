//
//  mobilenet-iOS.hpp
//  
//
//  Created by Rohan Kotwani on 6/25/17.
//  Copyright © 2017 Rohan Kotwani. All rights reserved.

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <stdio.h>


@interface mobileNetWrapper : NSObject

- (UIImage*) mobilenetWithOpenCV:(UIImage*)inputImage y_shift:(double)y_ x_shift:(double)x_ ;

@end
