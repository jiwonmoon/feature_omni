#ifndef SYSTEM_H
#define SYSTEM_H

#include "cv.hpp"
#include <iostream>

#include "Compare.h"

extern vector<float> avgRate_vec;
extern vector<int> avgN_vec;
extern vector<int> N1_vec;
extern vector<int> N2_vec;
extern vector<int> avgGoodMatch_vec;
extern vector<int> HammingDistance;

extern vector<double> TIME_avgGoodMatch_vec;
extern vector<double> TIME_avgDetect_vec;
extern vector<double> TIME_avgCompute_vec;
extern vector<double> TIME_DC1_vec;
extern vector<double> TIME_DC2_vec;


using namespace std;
using namespace cv;

namespace F_test
{
	class Compare;

	class System
	{
	public:
		enum S_type {
			PINHOLE = 0, OMNI = 1
		};
		enum F_type {
			ORB = 0, BRISK = 1, AKAZE = 2,
			ORB_EX = 3
		};

	public:
		System(const S_type sensor_type, const F_type feature_type, const string& strSettingsFile, int max_ImgNum, int _nFeatures);

		void TwoViewTest(const vector<Mat> img_vec, const Mat mask, std::vector<cv::Mat> LUT, const Mat cube_mask, int img_term, int delay);

		//convert fisheye image to cubemap
		void CvtFisheyeToCubeMap_reverseQuery(cv::Mat& cubemapImg, const cv::Mat& fisheyeImg);
		//creating mapping
		void CreateUndistortRectifyMap();
		void CreateUndistortRectifyMap_inv();
		//convert fisheye image to cubemap
		void CvtFisheyeToCubeMap(cv::Mat& cubemapImg, const cv::Mat& fisheyeImg);

		//convert fisheye image to cubemap
		void CvtCubeMapToFisheye(const cv::Mat& cubemapImg, cv::Mat& fisheyeImg);
		
		void print_RESULT();

		// image Mapping
		cv::Mat mMap1;
		cv::Mat mMap2;

		cv::Mat mMap1_inv;
		cv::Mat mMap2_inv;

	private:
		S_type cam_sensor;
		F_type typeOffeature;
		Compare* compare_;
		int max_imgN;



	};



}



#endif // !SYSTEM_H
