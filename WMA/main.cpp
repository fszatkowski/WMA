#include "Person.h"
#include "distance.h"

using namespace cv;
using namespace std;

//sources adress
string path = "C:/Users/fszat/source/repos/WMA/src/";

//different trained classifiers

//the best one
string face_cascade_name = path + "haarcascade_frontalface_alt.xml";

//faster, but not accurate
//string face_cascade_name = path + "lbpcascade_frontalface_improved.xml";

//worse than alt
//string face_cascade_name = path + "haarcascade_frontalface_alt2.xml";

//doesnt even get the profiles correctly
//string face_cascade_name = path + "haarcascade_profileface.xml";

//initialize cascade classifier
CascadeClassifier face_cascade;

//normalize the image depending on the number of its channels

double compareFaces(const cv::Mat &face_to_check, const cv::Mat &seeked_face)
{
	double similarity = 0;

	cv::Mat check = face_to_check.clone();
	cv::Mat seeked_f = seeked_face.clone();

	if (check.channels() != 1)
		cv::cvtColor(check, check, CV_BGR2GRAY);
	if (seeked_f.channels() != 1)
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

	similarity = double(1 - double(s) / double(max)) * 100;
	return similarity;
}

int main(int argc, char **argv)
{
	//load the cascade - if loading failed, throw an error
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "Error loading cascade" << std::endl;
		return -1;
	};

	//load the face we are seeking
	string file_name;
	file_name = "Tarantino.png";
	Mat face = imread(path + file_name);
	if (face.empty())
	{
		cout << "Error loading face" << std::endl;
		return -1;
	}
	face = norm_0_255(face);

	std::vector<Rect> face_seeked;

	//convert color and normalize
	Mat gray_face;
	cvtColor(face, gray_face, CV_BGR2GRAY);
	gray_face = norm_0_255(gray_face);

	//detect faces
	face_cascade.detectMultiScale(gray_face, face_seeked, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < face_seeked.size(); i++)
	{
		rectangle(face, Size(face_seeked[i].x, face_seeked[i].y), Size(face_seeked[i].x + face_seeked[i].width, face_seeked[i].y + face_seeked[i].height), Scalar(255, 0, 255), 2, 8);
	}

	Mat ROI(face, face_seeked[0]);
	Mat croppedFace;
	ROI.copyTo(croppedFace);

	//display loaded image with detected faces - for tests

	//imshow("Face_seeked", croppedFace);
	//waitKey(0);

	//load video
	string video_name;
	video_name = "3. - Desperados.mp4";
	VideoCapture capture(path + video_name);

	//actual frame
	Mat frame;

	//postprocessed frames stored 

	int frame_count = 0;

	Mat prev_frame;
	int prev_sum;
	int mem[5] = { 0, 0, 0, 0, 0 };
	int mean = 0;

	std::vector<Person> possible;

	while (true)
	{
		if (!capture.read(frame))
			break;

		std::vector<Rect> faces;

		Mat gray_frame;
		//convert color and normalize
		cvtColor(frame, gray_frame, CV_BGR2GRAY);
		gray_frame = norm_0_255(gray_frame);

		//check, if there is drastic change in the two frames
		//method - check if the difference in brightness of new and previous frame is bigger than 2 times mean obtained from previous 5 frames

		int s = 0;
		Mat difference;
		if (frame_count)
		{
			absdiff(gray_frame, prev_frame, difference);
		}
		else
		{
			absdiff(gray_frame, gray_frame, difference);
		}

		//sums the value of each pixel in the array
		s = sum(difference)[0];

		//obtain the mean of previous frames
		for (int i = 0; i < 5; i++)
		{
			mean += mem[i];
		}
		mean = mean / 5;

		//for the sake of simplicity, dont check first few frames - there should not be any change at the beggining anyway
		for (int i = 0; i < 4; i++)
		{
			mem[i] = mem[i + 1];
		}
		mem[4] = s;

		int brightness_change = abs(mean - s);

		bool frame_changed = false;

		if (brightness_change > 10 * mean && frame_count > 4)
			frame_changed = true;

		//test
		//std::cout << frame_count << " Suma pixeli: " << s << " Srednia poprzednich 5 pixeli: " << mean << " Roznica: " << brightness_change << " Zmiana? " << frame_changed << endl;

		//for faster processing, resize the image
		const int scale = 5;
		Mat resized_gray_frame(cvRound(gray_frame.rows / scale), cvRound(gray_frame.cols / scale), CV_8UC1);
		resize(gray_frame, resized_gray_frame, resized_gray_frame.size());

		//detect faces
		//used matrix, vector storing faces, scale factor, number of neighboors, ... , ..., minimum size of an face
		face_cascade.detectMultiScale(resized_gray_frame, faces, 1.05, 3, CV_HAAR_SCALE_IMAGE, Size(30, 30));

		//then change the position and size of detected faces, so that they match faces on the original image
		for (size_t i = 0; i < faces.size(); i++)
		{
			faces[i].x = scale * faces[i].x;
			faces[i].y = scale * faces[i].y;
			faces[i].width = scale * faces[i].width;
			faces[i].height = scale * faces[i].height;
		}

		//check if you can link detected faces to some existing person
		//for every detected face:
		for (size_t i = 0; i < faces.size(); i++)
		{
			//Rect face(faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Rect face(faces[i].x, faces[i].y, faces[i].width, faces[i].height);

			//get the matrix containing detected face
			Mat detected_face;
			Mat ROI(gray_frame, face);
			ROI.copyTo(detected_face);


			//for every existing person - check, if the face can belong to that person
			//if it cant - create a new one
			bool face_assigned = false;

			if (!frame_changed)
			{
				for (int j = 0; j < possible.size(); j++)
				{
					if (possible[j].add_if_close(face, frame_count, detected_face))
					{
						face_assigned = true;
						possible[j].compare(detected_face, croppedFace);
						break;
					}
				}
				if (!face_assigned)
				{
					Person new_person(frame_count, face);
					new_person.compare(detected_face, croppedFace);
					possible.push_back(new_person);
					break;
				}
			}
			else
			{
				int best_match = 0;
				int best_similarity = 0;
				for (int i = 0; i < possible.size(); i++)
				{
					double chance = compareFaces(possible[i].getPrev(), detected_face);
					if (chance > best_similarity)
					{
						best_match = i;
						best_similarity = chance;
					}
				}
				if (best_similarity > 60)
				{
					possible[best_match].add_if_matches(frame_count, detected_face, face);
					face_assigned = true;
					possible[best_match].compare(detected_face, croppedFace);
					break;
				}

				if (!face_assigned)
				{
					Person new_person(frame_count, face);
					new_person.compare(detected_face, croppedFace);
					possible.push_back(new_person);
					break;
				}
			}
		}

		//testing frame by frame 

		/*imshow("roznica", frame);
		char c = waitKey(0);
		if (c == 'q') break;
		*/
		frame_count++;

		std::cout << frame_count << " processing" << endl;

		prev_frame = gray_frame.clone();

	}

	Person bestperson = possible[0];
	double best_probability = bestperson.getProbability();

	for (int i = 1; i < possible.size(); i++)
	{
		if (possible[i].getProbability() > best_probability)
		{
			bestperson = possible[i];
		}
	}

	//reset the video	
	capture.release();

	std::cout << "Processing done" << endl;
	int x;
	cin >> x;

	VideoCapture final_capture(path + video_name);
	frame_count = 0;
	Scalar color = bestperson.getColor();
	std::cout << "Amount of found faces: " << possible.size() << endl;
	std::cout << "Highest match: " << bestperson.getProbability() << endl;


	while (final_capture.read(frame)) {
		if (bestperson.facesSize() > frame_count)
			rectangle(frame, bestperson.getFace(frame_count), color, 2, 8);
		imshow("result", frame);
		char c = waitKey(20);
		if (c == 'q')
		{
			return 0;
		}
		frame_count++;
	}

	return 0;
}