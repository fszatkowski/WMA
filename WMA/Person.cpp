#include "Person.h"
#include "math.h"

Person::Person(int frame_count, cv::Rect face)
{
	seeked_face_probability = 0;
	while (faces.size() != frame_count)
	{
		cv::Rect empty(0, 0, 0, 0);
		faces.push_back(empty);
	}
	faces.push_back(face);
	color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

Person::Person()
{
}

Person::~Person()
{
}

cv::Rect Person::getFace(int frame)
{
	return faces.at(frame);
}

cv::Scalar Person::getColor()
{
	return color;
}

cv::Mat Person::getPrev()
{
	return previous_face;
}

double Person::getProbability()
{
	return seeked_face_probability;
}

/*void Person::setBest(const cv::Mat & new_best, const double & prob)
{
	best_detected_face = new_best;
	best_detected_face_probability = prob;
}*/

bool Person::add_if_close(const cv::Rect & checked, int frame_count, const cv::Mat &face_to_check)
{
	//if the position of the face detected on the frame is close to the previous position of person's face,
	//add it to the container of the person's faces
	//!only use this, if the scene doesnt change drastically

	//possible problem - face not detected on the previous image makes the algorith crash
	//solution - check few previous faces, if they match, then fill the container and add this one as well

	//solution may cause some problems, if new person will be detected in the same place - however it may not be problematic depending on how do the frames change

	if (abs(checked.x - faces.at(faces.size()-1).x) + abs(checked.y - faces.at(faces.size() - 1).y) <= (faces.at(faces.size() - 1).width + faces.at(faces.size() - 1).height) / 20)
	{
		faces.push_back(checked);
		previous_face = face_to_check.clone();
		return 1;
	}

	else if (faces.size() >=2 && abs(checked.x - faces.at(faces.size() - 2).x) + abs(checked.y - faces.at(faces.size() - 2).y) <= (faces.at(faces.size() - 2).width + faces.at(faces.size() - 2).height) / 20)
	{
		faces.at(faces.size() - 1) = checked;
		faces.push_back(checked);
		previous_face = face_to_check.clone();
		return 1;
	}

	else if (faces.size() >= 3 && abs(checked.x - faces.at(faces.size() - 3).x) + abs(checked.y - faces.at(faces.size() - 3).y) <= (faces.at(faces.size() - 3).width + faces.at(faces.size() - 3).height) / 20)
	{
		faces.at(faces.size() - 2) = checked;
		faces.at(faces.size() - 1) = checked;
		faces.push_back(checked);
		previous_face = face_to_check.clone();
		return 1;
	}
	else return 0;
}

void Person::add_if_matches(int frame_count, const cv::Mat &face_to_check, const cv::Rect &face_coords)
{
	//compare the face detected with the best face of this person
	//if they match, capture this face as person's face and fill the faces in the vector up this one with empty faces

	while (faces.size() != frame_count)
	{
		cv::Rect empty(0, 0, 0, 0);
		faces.push_back(empty);
	}
	faces.push_back(face_coords);
	previous_face = face_to_check.clone();

}

double Person::match(const cv::Mat &face_to_check)
{
	double similarity = 0;

	cv::Mat check = face_to_check.clone();
	
	resize(check, check, previous_face.size());
	
	threshold(previous_face, previous_face, 20, 255, cv::THRESH_BINARY);
	morphologyEx(previous_face, previous_face, cv::MORPH_OPEN, cv::Mat());

	threshold(check, check, 20, 255, cv::THRESH_BINARY);
	morphologyEx(check, check, cv::MORPH_OPEN, cv::Mat());

	cv::Mat difference;
	absdiff(previous_face, check, difference);

	int max = previous_face.cols * previous_face.rows * 255;
	int s = sum(difference)[0];
	similarity = double(1 - double(s) / double(max)) * 100;

	return similarity;
}

void Person::compare(const cv::Mat &face_to_check, const cv::Mat &seeked_face)
{
	double probability = 0;
	
	cv::Mat check = face_to_check.clone();
	cv::Mat seeked_f = seeked_face.clone();

	if(check.channels()!=1)
		cv::cvtColor(check, check, CV_BGR2GRAY);
	if(seeked_f.channels()!=1)
		cv::cvtColor(seeked_f, seeked_f, CV_BGR2GRAY);

	resize(check, check, seeked_f.size());

	threshold(seeked_f, seeked_f, 20, 255, cv::THRESH_BINARY);
	morphologyEx(seeked_f, seeked_f, cv::MORPH_OPEN, cv::Mat());

	threshold(check, check, 20, 255, cv::THRESH_BINARY);
	morphologyEx(check, check, cv::MORPH_OPEN, cv::Mat());

	cv::Mat facesDifference;
	absdiff(seeked_f, check, facesDifference);

	int max = seeked_f.cols * seeked_f.rows * 255;
	int s = sum(facesDifference)[0];
	probability = double(1 - double(s) / double(max)) * 100;
	
	if (probability > seeked_face_probability) {
		seeked_face_probability = probability;
	}
}

int Person::facesSize()
{
	return faces.size();
}