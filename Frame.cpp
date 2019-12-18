#include "Frame.h"
#include <time.h>

extern vector<double> TIME_avgDetect_vec;
extern vector<double> TIME_avgCompute_vec;

namespace F_test
{
	//float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;


	Frame::Frame() {}
	Frame::Frame(const Mat img, const Mat mask, int feature_type)
	{
		F_img = img.clone();


		switch (feature_type)
		{
		case 0://ORB
			Extract_ORB(img, mask);///
			break;
		case 1://BRISK
			Extract_BRISK(img, mask);///
			break;
		case 2://AKAZE
			Extract_AKAZE(img, mask);///
			break;

		default:
			cerr << "!!! No adequate feature_type err !!!" << endl;
			exit(0);
			break;
		}

		N = mvKeys.size();
	}

	Frame::Frame(const Mat img, const Mat mask, const Mat cube_mask, ORBextractor* ORBextract)
	{
		F_img = img.clone();

		int offset = 0;
		int width = CamModelGeneral::GetCamera()->GetCubeFaceWidth(), height = CamModelGeneral::GetCamera()->GetCubeFaceHeight();
		cv::Mat cubemapImg(height * 3, width * 3, CV_8U, cv::Scalar::all(0));
		cv::Mat cubemapImg_front = cubemapImg.rowRange(height, 2 * height).colRange(width, 2 * width);
		cv::Mat cubemapImg_left = cubemapImg.rowRange(height, 2 * height).colRange(0 + offset, width + offset);
		cv::Mat cubemapImg_right = cubemapImg.rowRange(height, 2 * height).colRange(2 * width - offset, 3 * width - offset);
		cv::Mat cubemapImg_upper = cubemapImg.rowRange(0 + offset, height + offset).colRange(width, 2 * width);
		cv::Mat cubemapImg_lower = cubemapImg.rowRange(2 * height - offset, 3 * height - offset).colRange(width, 2 * width);

		//F_system.CvtFisheyeToCubeMap_reverseQuery_withInterpolation(cubemapImg, img_set[ni], cv::INTER_LINEAR);
		CvtFisheyeToCubeMap_reverseQuery(cubemapImg, F_img);
		C_img = cubemapImg;


		mpORBextractor = ORBextract;

		// ORB extraction
		Extract_ORB_EX(img, mask, cube_mask);

		N = mvKeys.size();

	}


	/*
	https://docs.opencv.org/3.4.1/db/d95/classcv_1_1ORB.html
	*/
	void Frame::Extract_ORB(Mat img, Mat mask)
	{
		Ptr<ORB> orbF = ORB::create(3000);
		//orbF->detectAndCompute(img, mask, mvKeys, mDescriptors);

		clock_t start_detect = clock();
		orbF->detect(img, mvKeys, mask);///
		double duration_detect = (double)(clock() - start_detect);

		clock_t start_compute = clock();
		orbF->compute(img, mvKeys, mDescriptors);///
		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);
		TIME_avgCompute_vec.push_back(duration_compute);
	}

	/*
	https://docs.opencv.org/3.4.1/de/dbf/classcv_1_1BRISK.html
	*/
	void Frame::Extract_BRISK(Mat img, Mat mask)
	{
		Ptr<BRISK> briskF = BRISK::create();
		//briskF->detectAndCompute(img, mask, mvKeys, mDescriptors);

		clock_t start_detect = clock();
		briskF->detect(img, mvKeys, mask);///
		double duration_detect = (double)(clock() - start_detect);

		clock_t start_compute = clock();
		briskF->compute(img, mvKeys, mDescriptors);///
		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);
		TIME_avgCompute_vec.push_back(duration_compute);
	}

	/*
	https://docs.opencv.org/3.4.1/d8/d30/classcv_1_1AKAZE.html
	*/
	void Frame::Extract_AKAZE(Mat img, Mat mask)
	{

		Ptr<AKAZE> akazeF = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT);//DESCRIPTOR_MLDB_UPRIGHT	DESCRIPTOR_MLDB
		//akazeF->detectAndCompute(img, mask, mvKeys, mDescriptors);

		clock_t start_detect = clock();
		akazeF->detect(img, mvKeys, mask);///
		double duration_detect = (double)(clock() - start_detect);

		clock_t start_compute = clock();
		akazeF->compute(img, mvKeys, mDescriptors);
		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);///
		TIME_avgCompute_vec.push_back(duration_compute);
	}

	void Frame::Extract_ORB_EX(Mat img, Mat mask, const Mat cube_mask)
	{

		if (img.channels() != 1)
			cvtColor(img, img, CV_BGR2GRAY);

		///detection part
		clock_t start_detect = clock();

		// ORB extraction
		(*mpORBextractor)(img, mask, cube_mask, mvKeys, mDescriptors);

		double duration_detect = (double)(clock() - start_detect);

		///destriptor part
		clock_t start_compute = clock();


		double duration_compute = (double)(clock() - start_compute);

		TIME_avgDetect_vec.push_back(duration_detect);///
		TIME_avgCompute_vec.push_back(duration_compute);
	}

	void Frame::CvtFisheyeToCubeMap_reverseQuery(cv::Mat& cubemapImg, const cv::Mat& fisheyeImg)
	{
		//clear rectified image
		cubemapImg.setTo(cv::Scalar::all(0));
		int width3 = CamModelGeneral::GetCamera()->GetCubeFaceWidth() * 3, height3 = CamModelGeneral::GetCamera()->GetCubeFaceHeight() * 3;
		int Iw = CamModelGeneral::GetCamera()->GetFisheyeWidth(), Ih = CamModelGeneral::GetCamera()->GetFisheyeHeight();
		for (int i = 0; i < width3; ++i)
		{
			for (int j = 0; j < height3; ++j)
			{
				double u, v;
				CamModelGeneral::GetCamera()->CubemapToFisheye(u, v, static_cast<double>(i), static_cast<double>(j));
				//int ui = cvRound(u), vi = cvRound(v);
				int ui = std::floor(u), vi = std::floor(v);
				// on some face but doesn't map to a fisheye valid region
				if (ui < 0 || vi < 0 || ui >= Iw || vi >= Ih)
					continue;

				uchar intensity = fisheyeImg.at<uchar>(vi, ui);
				cubemapImg.at<uchar>(j, i) = intensity;
			}
		}
	}

}
