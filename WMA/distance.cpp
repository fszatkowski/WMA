#include "distance.h"

static cv::Mat norm_0_255(cv::InputArray _src) {
	cv::Mat src = _src.getMat();
	// Create and return normalized image:
	cv::Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

//face1 is the face we check, face2 is the face that we are seeking
double fourierDistance(const cv::Mat & face1, const cv::Mat & face2)
{
	//work on clones
	cv::Mat to_check = face1.clone();
	cv::Mat second = face2.clone();

	//check if we are working on grayscale - if not, convert
	if (to_check.channels() != 1)
		cv::cvtColor(to_check, to_check, CV_BGR2GRAY);
	if (second.channels() != 1)
		cv::cvtColor(second, second, CV_BGR2GRAY);

	//resize the images to the same size using cubic interpolation
	resize(to_check, to_check, second.size(), cv::INTER_CUBIC);

	//normalize histograms
	norm_0_255(face1);
	norm_0_255(face2);

	//SOME MORE PREPROCESSING?


}
