#ifndef  FRMAE_H
#define FRMAE_H

#include "cv.hpp"
#include <iostream>

#include "ORBextractor.h"
#include "CamModelGeneral.h"

using namespace std;
using namespace cv;

namespace F_test
{
	class Compare;
	class ORBextractor;

	class Frame {
	public:
		Frame();
		Frame(const Mat img, const Mat mask, int feature_type);
		Frame(const Mat img, const Mat mask, const Mat cube_mask, ORBextractor* ORBextract);


		void Extract_ORB(Mat img, Mat mask);///
		void Extract_BRISK(Mat img, Mat mask);///
		void Extract_AKAZE(Mat img, Mat mask);///
		void Extract_ORB_EX(Mat img, Mat mask, const Mat cube_mask);
		void Extract_ORB_EX_1(Mat img, Mat mask, const Mat cube_mask);

		void CvtFisheyeToCubeMap_reverseQuery(cv::Mat& cubemapImg, const cv::Mat& fisheyeImg);


	public:
		ORBextractor* mpORBextractor;
		
		Mat F_img;
		Mat C_img;
		//num of keypoints
		int N;
		// ORB feature point
		std::vector<cv::KeyPoint> mvKeys;
		// ORB descriptor, each row associated to a keypoint.
		cv::Mat mDescriptors;



	private:



	};





}



#endif // ! FRMAE_H
