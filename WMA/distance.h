#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

//calculating the histograms for gray and hsv face
cv::Mat gray_hist(const cv::Mat &frame, const cv::Rect &face, const cv::Size &size);
cv::Mat hsv_hist(const cv::Mat &frame, const cv::Rect &face, const cv::Size &size);

//earth mover distance
double distance_emd(const cv::Mat &frame, const cv::Rect &face, const cv::Mat &hsv_histogram, const cv::Size &size);

//calculate if the scenery has changed
bool scene_change(const cv::Mat &frame, const cv::Mat previous_frame);