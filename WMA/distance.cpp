#include "distance.h"

using namespace cv;

cv::Mat gray_hist(const cv::Mat &frame, const cv::Rect &face, const cv::Size &size)
{
	cv::Mat h = frame(face).clone();
	cv::cvtColor(h, h, CV_BGR2GRAY);
	resize(h, h, size, CV_INTER_CUBIC);
	
	cv::Mat hist;

	//only one channel - gray
	int channels[] = { 0 };
	//quantization
	int histSize[] = { 256 };
	//range of each level
	float cranges[] = { 0, 256 };
	const float* ranges[] = { cranges };

	calcHist(&h, 1, channels, Mat(), 
		hist, 1, histSize, ranges, 
		true, 
		false);
	//normalize(hist, hist, 0, 1, CV_MINMAX);

	return hist;
}

cv::Mat hsv_hist(const cv::Mat &frame, const cv::Rect &face, const cv::Size &size)
{
	cv::Mat h = frame(face).clone();
	cv::cvtColor(h, h, CV_BGR2HSV);
	resize(h, h, size, CV_INTER_CUBIC);
	
	cv::Mat hist;

	//hue and saturation quantisation
	int hbins = 30, sbins = 32;
	int channels[] = { 0,  1 };
	int histSize[] = { hbins, sbins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 255 };
	const float* ranges[] = { hranges, sranges };

	calcHist(&h, 1, channels, Mat(), // do not use mask   
		hist, 2, histSize, ranges,
		true, // the histogram is uniform   
		false);
	normalize(hist, hist, 0, 1, CV_MINMAX);

	return hist;
}

double distance_emd(const cv::Mat &frame, const cv::Rect &face, const cv::Mat &hsv_histogram, const cv::Size &size)
{
	//work on clones
	cv::Mat to_check = hsv_hist(frame, face, size);
	cv::Mat model_histogram = hsv_histogram.clone();

	//make signatures
	int hbins = 30, sbins = 32;
	int channels[] = { 0,  1 };
	int histSize[] = { hbins, sbins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 255 };
	const float* ranges[] = { hranges, sranges };
	int numrows = hbins * sbins;

	Mat sig1(numrows, 3, CV_32FC1);
	Mat sig2(numrows, 3, CV_32FC1);

	//fill value into signature
	for (int h = 0; h < hbins; h++)
	{
		for (int s = 0; s < sbins; ++s)
		{
			float binval = model_histogram.at<float>(h, s);
			sig1.at<float>(h*sbins + s, 0) = binval;
			sig1.at<float>(h*sbins + s, 1) = h;
			sig1.at<float>(h*sbins + s, 2) = s;

			binval = to_check.at<float>(h, s);
			sig2.at<float>(h*sbins + s, 0) = binval;
			sig2.at<float>(h*sbins + s, 1) = h;
			sig2.at<float>(h*sbins + s, 2) = s;
		}
	}

	//return similarity of 2 images
	//0 means identical

	double distance = cv::EMD(sig1, sig2, CV_DIST_L2);
	std::cout << distance << std::endl;

	return distance; 
}

bool scene_change(const cv::Mat & frame, const cv::Mat previous_frame)
{
	if (previous_frame.empty())
		return 1;
	
	Mat difference;
	absdiff(frame, previous_frame, difference);

	threshold(difference, difference, 150, 255, THRESH_BINARY);
	int value = sum(difference)[0];
	//std::cout << value << std::endl;
	//std::cout << frame.cols * frame.rows << std::endl;

	if (value > frame.cols * frame.rows * 30)
	{
		return 1;
	}
	else 
	return 0;
}
