#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

//normalization
static cv::Mat norm_0_255(cv::InputArray _src);

//return distance
double fourierDistance(const cv::Mat &face1, const cv::Mat &face2);

