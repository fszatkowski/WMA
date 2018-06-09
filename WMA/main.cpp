#include "distance.h"

using namespace cv;
using namespace std;

//sources adress
string path = "C:/Users/fszat/source/repos/WMA/src/";

//initialize cascade classifier - a class for object detection
//use pretrained opencv haar cascade
string face_cascade_name = path + "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

int main(int argc, char **argv)
{
	//INITIALIZING CLASSIFIER, LOADING AND PREPARING IMAGE AND VIDEO



	//load the cascade - if loading failed, throw an error
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "Error loading cascade" << std::endl;
		return -1;
	};

	//load the face we are seeking - -||-
	string file_name = "Tarantino.png";
	Mat base_face = imread(path + file_name);
	if (base_face.empty())
	{
		cout << "Error loading face" << std::endl;
		return -1;
	}

	//extract the face from image
	//convert the basic image to grayscale and use classifier

	std::vector<Rect> face_seeked;
	Mat gray_base_face;
	cvtColor(base_face, gray_base_face, CV_BGR2GRAY);

	//detect faces
	//returns list of rectangles containing detected faces
	//detects objects of different sizes
	//1.05 - how much is the image scaled
	//3 - minimum neighboors detected to keep processing face
	//flags - not used
	//minimum size of detected face
	face_cascade.detectMultiScale(gray_base_face, face_seeked, 1.05, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
	
	//draw rectangle on detected face
	for (size_t i = 0; i < face_seeked.size(); i++)
	{
		rectangle(base_face, Size(face_seeked[i].x, face_seeked[i].y), Size(face_seeked[i].x + face_seeked[i].width, face_seeked[i].y + face_seeked[i].height), Scalar(255, 0, 255), 2, 8);
	}

	//if there is no face throw and error
	if (face_seeked.size() == 0)
	{
		cout << "Given image contains no face" << std::endl;
		return -1;
	}

	//save the detected face, which is the only one detected
	Size faceSize = face_seeked[0].size();

	//get HSV histogram for later use
	Mat hsv_model_hist = hsv_hist(base_face, face_seeked[0], base_face.size());

	//load video
	string video_name;
	video_name = "3. - DesperadoTrim.mp4";
	VideoCapture capture(path + video_name);

	//actual frame
	Mat frame;

	//and some data from previous frames
	//to track faces better
	Mat previous_frame;
	Rect previous_face;
	bool frame_filled = true;
	bool do_emd = false;

	while (true)
	{
		//keep capturing frame by frame
		if (!capture.read(frame))
		{
			break;
		}


		//FACE DETECTION ON ACTUAL FRAME

		//convert color to greyscale for cascade classifier
		Mat gray_frame;
		cvtColor(frame, gray_frame, CV_BGR2GRAY);
		equalizeHist(gray_frame, gray_frame);

		//work on smaller image for faster detection
		const int scale = 3;
		Mat resized_gray_frame(cvRound(gray_frame.rows / scale), cvRound(gray_frame.cols / scale), CV_8UC1);
		resize(gray_frame, resized_gray_frame, resized_gray_frame.size());

		//detect faces using Haar cascade classifier
		std::vector<Rect> faces_detected;
		face_cascade.detectMultiScale(resized_gray_frame, faces_detected, 1.2, 3, CV_HAAR_SCALE_IMAGE, Size(30, 30));

		//then change the position and size of detected faces, so that they match faces on the original image
		for (size_t i = 0; i < faces_detected.size(); i++)
		{
			faces_detected[i].x = scale * faces_detected[i].x;
			faces_detected[i].y = scale * faces_detected[i].y;
			faces_detected[i].width = scale * faces_detected[i].width;
			faces_detected[i].height = scale * faces_detected[i].height;
		}




		//DECIDE WHAT TO SHOW

		int best_match = 0;

		//if scene changed, use EMD to decide which face is the one we seek
		if (scene_change(gray_frame, previous_frame) || do_emd)
		{
			int best_match = 0;
			double best_distance;

			//if there were faces, preform emd
			if (faces_detected.size() != 0) {
				best_distance = distance_emd(frame, faces_detected[0], hsv_model_hist, faceSize);
				if (faces_detected.size() > 1) {
					for (size_t i = 0; i < faces_detected.size(); i++)
					{
						double tempDistance = distance_emd(frame, faces_detected[i], hsv_model_hist, faceSize);
						if (tempDistance < best_distance)
						{
							best_distance = tempDistance;
							best_match = i;
						}
					}
				}		
				rectangle(frame, faces_detected[best_match], (255, 0, 255), 4, 8);
				previous_face = faces_detected[best_match];
				do_emd = false;
				//cout << "scene change emd" << endl;
			}
			//else use a flag to preform emd again until we detect something
			else
			{
				do_emd = true;
				//cout << "emergency emd" << endl;
			}
		}
		//if scene didnt change, follow the most likely one until the scene changes
		else
		{
			//if there were faces detected, check their coordinates
			if (faces_detected.size() != 0)
			{
				int distance = abs(faces_detected[0].x - previous_face.x) + abs(faces_detected[0].y - previous_face.y);
				best_match = 0;
				for (int i = 0; i < faces_detected.size(); i++) 
				{
					if (abs(faces_detected[i].x - previous_face.x) + abs(faces_detected[i].y - previous_face.y) < distance)
					{
						distance = abs(faces_detected[i].x - previous_face.x) + abs(faces_detected[i].y - previous_face.y);
						best_match = i; 
					}
				}
				//chose the closest one as the face IF the face is close enough
				if (!previous_face.empty() && distance < previous_face.x + previous_face.y / 10	)
				{
					rectangle(frame, faces_detected[best_match], (255, 0, 255), 4, 8);
					previous_face = faces_detected[best_match];
					//cout << "chosing closest" << endl;
				}
				//else fill for one frame with the last detected face
				else
				{
					rectangle(frame, previous_face, (255, 0, 255), 4, 8);
					Rect empty;
					previous_face = empty;
					//cout << "filling cause detected faces were too far" << endl;
				}
			}
			//fill once if there are no faces to keep contionious flow of images
			else if (faces_detected.size() == 0 && !frame_filled)
			{
				rectangle(frame, previous_face, (255, 0, 255), 4, 8);
				previous_face = previous_face;
				frame_filled = true;
				//cout << "filling cause no detected faces" << endl;
			}
			else if (faces_detected.size() == 0 && frame_filled)
			{
				frame_filled = false;
				Rect empty;
				previous_face = empty;
				do_emd = true;
				//cout << "lost track of face" << endl;
			}
		}


		//RESULT


		//show result
		imshow("faceDetected", frame);
		char c = waitKey(20);
		if (c == 'q') break;

		previous_frame = gray_frame;

	}
	return 0;
}