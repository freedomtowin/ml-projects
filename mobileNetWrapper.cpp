//
//  mobilenet-iOS.cpp
//  
//
//  Created by Rohan Kotwani on 6/25/17.
//  Copyright © 2017 Rohan Kotwani. All rights reserved.


#include <opencv2/opencv.hpp>
#include "OpenCVWrapper.h"
#import "UIImage+OpenCV.h" // See below to create this


//#import <opencv2/imgcodecs/ios.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

#include <iterator>
#include <vector>

@implementation mobileNetWrapper{
    cv::dnn::Net net;
}

const std::vector<std::string> classNames = {
    "background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
};

- (instancetype) init {
    cv::String pathProto = [[[NSBundle mainBundle] pathForResource:@"MobileNetSSD_deploy" ofType:@"prototxt"] UTF8String];
    cv::String pathModel = [[[NSBundle mainBundle] pathForResource:@"MobileNetSSD_deploy" ofType:@"caffemodel"] UTF8String];
    
    self->net = cv::dnn::readNetFromCaffe(pathProto, pathModel);
    
    return self;
}
    //    NSMutableArray *arr = [NSMutableArray array];
    
    for (int i = 0; i < 5; i++) {
        [arr addObject:[[NSNumber numberWithDouble:return_arr[i]] init]];
    }
    
    return arr;
    //    return (shift[1]>=0)*10000000+abs(shift[1])*10000 + (shift[0]>=0)*1000+ abs(shift[0]);
    
}


cv::Mat cvMatMakeRgb(UIImage const* uiImage) {
    cv::Mat cvMat;
    //    UIImageToMat(uiImage, cvMat, true);
    cvMat = [uiImage  CVMat];
    if (cvMat.channels() == 4) cvtColor(cvMat, cvMat, CV_BGRA2BGR);
    
    return cvMat;
}

- (UIImage*) mobilenetWithOpenCV:(UIImage*)inputImage y_shift:(double)y_  x_shift:(double)x_ {
    
    
    Mat mat = cvMatMakeRgb(inputImage);
    
    string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"};
    
    //    cv::String pathProto = [[[NSBundle mainBundle] pathForResource:@"MobileNetSSD_deploy" ofType:@"prototxt"] UTF8String];
    //    cv::String pathModel = [[[NSBundle mainBundle] pathForResource:@"MobileNetSSD_deploy" ofType:@"caffemodel"] UTF8String];
    
    //    Net net = dnn::readNetFromCaffe(pathProto, pathModel);
    Mat img2;
    resize(mat, img2, cv::Size(300,300));
    cv::Mat inputBlob = dnn::blobFromImage(img2, 1.0/127.5, cv::Size(300,300), 127.5, false);
    //    cout << inputBlob.size << endl;
    self->net.setInput(inputBlob, "data");
    cv::Mat detection = self->net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    
    ostringstream ss;
    float confidenceThreshold = 0.2;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        
        if (confidence > confidenceThreshold)
        {
            int idx = static_cast<int>(detectionMat.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * mat.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * mat.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * mat.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * mat.rows);
            
            cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));
            
            cv::rectangle(mat, object, Scalar(0, 255, 0), 2);
            
            cout << CLASSES[idx] << ": " << confidence << endl;
            
            ss.str("");
            ss << confidence;
            String conf(ss.str());
            String label = CLASSES[idx] + ": " + conf;
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(mat, label, cv::Point(xLeftBottom-5, yLeftBottom-5),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0));
            cout << label << endl;
            }
            }
            
            return [UIImage imageWithCVMat:mat];
        
        @end
