#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

//each person is represented by the vector of rectangles that contain position of its faces
//frame corresponds to the position on the vector
//if person doesnt have a face on the frame no. i, faces[i] is Rect(0, 0, 0, 0)
//each person has its color and contains the most accurate face ever detected for it
//at the end of processing, the best face detected is compared with seeked face and given the probability of matching
//the person with the highest probability is given the label "seeked face" and tracked in the video

class Person
{
private:
	std::string label;
	cv::Scalar color;
	std::vector<cv::Rect> faces;
	cv::Mat previous_face;
	double seeked_face_probability;
public:
	Person(int frame_count, cv::Rect face);
	Person();
	~Person();
	cv::Rect getFace(int frame);
	cv::Scalar getColor();
	cv::Mat getPrev();
	int facesSize();
	double getProbability();
	void compare(const cv::Mat &face_to_check, const cv::Mat &seeked_f);
	bool add_if_close(const cv::Rect &checked, int frame_count, const cv::Mat &face_to_check);
	void add_if_matches(int frame_count, const cv::Mat &face_to_check, const cv::Rect &face_coords);
	double match(const cv::Mat &face_to_check);
};
