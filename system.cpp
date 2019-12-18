#include "system.h"
#include "CamModelGeneral.h"

namespace F_test
{
	/* main Feature test system */
	System::System(const S_type sensor_type, const F_type feature_type, const string& strSettingsFile, int max_ImgNum, int _nFeatures)
	{
		cout << "[SYSTEM:: Feature TEST, Use of (" << feature_type << ") type feature(0: ORB, 1: BRISK, 2: AKAZE, 3: ORB_EX), 12/11 version]" << endl << endl;


		//Check settings file
		cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
		if (!fsSettings.isOpened())
		{
			cerr << "Failed to open settings file at: " << strSettingsFile << endl;
			exit(EXIT_FAILURE);
		}

		//read in params
		int nrpol = fsSettings["Camera.nrpol"];
		int nrinvpol = fsSettings["Camera.nrinvpol"];

		cv::Mat_<double> poly = cv::Mat::zeros(5, 1, CV_64F);
		for (int i = 0; i < nrpol; ++i)
			poly.at<double>(i, 0) = fsSettings["Camera.a" + std::to_string(i)];
		cv::Mat_<double>  invpoly = cv::Mat::zeros(12, 1, CV_64F);
		for (int i = 0; i < nrinvpol; ++i)
			invpoly.at<double>(i, 0) = fsSettings["Camera.pol" + std::to_string(i)];

		int Iw = (int)fsSettings["Camera.Iw"];
		int Ih = (int)fsSettings["Camera.Ih"];

		double cdeu0v0[5] = { fsSettings["Camera.c"], fsSettings["Camera.d"], fsSettings["Camera.e"],
			fsSettings["Camera.u0"], fsSettings["Camera.v0"] };

		//cubemap params
		int nFaceH = fsSettings["CubeFace.h"];
		int nFaceW = fsSettings["CubeFace.w"];
		double fx = static_cast<double>(nFaceW) / 2, fy = static_cast<double>(nFaceH) / 2;
		double cx = static_cast<double>(nFaceW) / 2, cy = static_cast<double>(nFaceH) / 2;

		double camFov = fsSettings["Camera.fov"];


		//Set camera model
		CamModelGeneral::GetCamera()->SetCamParams(cdeu0v0, poly, invpoly, Iw, Ih, fx, fy, cx, cy, nFaceW, nFaceH, camFov);
		std::cout << "finish creating general camera model" << std::endl;

		//Create mapping from cubemap to fisheye
		CreateUndistortRectifyMap();

		cam_sensor = sensor_type;//tpye of sensor
		typeOffeature = feature_type;//type of feature


		float fScaleFactor = 1.2;
		int nLevels = 8;
		int fIniThFAST = 20;
		int fMinThFAST = 7;
		compare_ = new Compare(_nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

		max_imgN = max_ImgNum;//last img vector idx
	}

	void System::TwoViewTest(const vector<Mat> img_vec, const Mat mask, const Mat cube_mask, int img_term, int delay)
	{

		for (int ni = 0; ni < (img_vec.size() - img_term); ni++)
		{
			compare_->compare2img(img_vec, ni, ni + img_term, mask, cube_mask, typeOffeature);
			waitKey(delay);
		}

	}




	//convert fisheye image to cubemap
	void System::CvtFisheyeToCubeMap_reverseQuery(cv::Mat& cubemapImg, const cv::Mat& fisheyeImg)
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



	void System::CreateUndistortRectifyMap()
	{
		int width3 = CamModelGeneral::GetCamera()->GetCubeFaceWidth() * 3, height3 = CamModelGeneral::GetCamera()->GetCubeFaceHeight() * 3;
		//create map for u(mMap1) and v(mMap2)
		mMap1.create(height3, width3, CV_32F);
		mMap2.create(height3, width3, CV_32F);
		mMap1.setTo(cv::Scalar::all(0));
		mMap2.setTo(cv::Scalar::all(0));

		int Iw = CamModelGeneral::GetCamera()->GetFisheyeWidth(), Ih = CamModelGeneral::GetCamera()->GetFisheyeHeight();
		for (int y = 0; y < height3; ++y)
		{
			for (int x = 0; x < width3; ++x)
			{
				double u, v;
				CamModelGeneral::GetCamera()->CubemapToFisheye(u, v, static_cast<double>(x), static_cast<double>(y));
				// on some face but doesn't map to a fisheye valid region
				if (u < 0 || v < 0 || u >= Iw || v >= Ih)
					continue;
				mMap1.at<float>(y, x) = static_cast<float>(u);
				mMap2.at<float>(y, x) = static_cast<float>(v);
			}
		}
	}


	//convert fisheye image to cubemap
	void System::CvtFisheyeToCubeMap_reverseQuery_withInterpolation(cv::Mat& cubemapImg, const cv::Mat& fisheyeImg, int interpolation, int borderType, const cv::Scalar& borderValue)
	{

		const int offset = 0;
		const int width = CamModelGeneral::GetCamera()->GetCubeFaceWidth(), height = CamModelGeneral::GetCamera()->GetCubeFaceHeight();

		//front
		cv::Mat cubemapImg_front = cubemapImg.rowRange(height, 2 * height).colRange(width, 2 * width);
		cv::Mat mMap1_front = mMap1.rowRange(height, 2 * height).colRange(width, 2 * width);
		cv::Mat mMap2_front = mMap2.rowRange(height, 2 * height).colRange(width, 2 * width);
		cv::remap(fisheyeImg, cubemapImg_front, mMap1_front, mMap2_front, interpolation, borderType, borderValue);

		//left
		cv::Mat cubemapImg_left = cubemapImg.rowRange(height, 2 * height).colRange(0 + offset, width + offset);
		cv::Mat mMap1_left = mMap1.rowRange(height, 2 * height).colRange(0 + offset, width + offset);
		cv::Mat mMap2_left = mMap2.rowRange(height, 2 * height).colRange(0 + offset, width + offset);
		cv::remap(fisheyeImg, cubemapImg_left, mMap1_left, mMap2_left, interpolation, borderType, borderValue);

		//right
		cv::Mat cubemapImg_right = cubemapImg.rowRange(height, 2 * height).colRange(2 * width - offset, 3 * width - offset);
		cv::Mat mMap1_right = mMap1.rowRange(height, 2 * height).colRange(2 * width - offset, 3 * width - offset);
		cv::Mat mMap2_right = mMap2.rowRange(height, 2 * height).colRange(2 * width - offset, 3 * width - offset);
		cv::remap(fisheyeImg, cubemapImg_right, mMap1_right, mMap2_right, interpolation, borderType, borderValue);

		//upper
		cv::Mat cubemapImg_upper = cubemapImg.rowRange(0 + offset, height + offset).colRange(width, 2 * width);
		cv::Mat mMap1_upper = mMap1.rowRange(0 + offset, height + offset).colRange(width, 2 * width);
		cv::Mat mMap2_upper = mMap2.rowRange(0 + offset, height + offset).colRange(width, 2 * width);
		cv::remap(fisheyeImg, cubemapImg_upper, mMap1_upper, mMap2_upper, interpolation, borderType, borderValue);

		//lower
		cv::Mat cubemapImg_lower = cubemapImg.rowRange(2 * height - offset, 3 * height - offset).colRange(width, 2 * width);
		cv::Mat mMap1_lower = mMap1.rowRange(2 * height - offset, 3 * height - offset).colRange(width, 2 * width);
		cv::Mat mMap2_lower = mMap2.rowRange(2 * height - offset, 3 * height - offset).colRange(width, 2 * width);
		cv::remap(fisheyeImg, cubemapImg_lower, mMap1_lower, mMap2_lower, interpolation, borderType, borderValue);
		//interpolation with cv::remap for each face
	}



	void System::print_RESULT()
	{
		cout << "[ ========== RESULT ========== ]" << endl;
		float avgRate = 0;
		int avgN = 0;
		int avgGoodMatch = 0;

		double TIME_avgGoodMatch = 0;
		double TIME_avgDetect = 0;
		double TIME_avgCompute = 0;

		for (int i = 0; i < avgRate_vec.size(); i++)
		{
			avgRate += avgRate_vec[i];
			avgN += avgN_vec[i];
			avgGoodMatch += avgGoodMatch_vec[i];

			TIME_avgGoodMatch += TIME_avgGoodMatch_vec[i];
			TIME_avgDetect += TIME_avgDetect_vec[i];
			TIME_avgCompute += TIME_avgCompute_vec[i];
		}
		avgRate /= avgRate_vec.size();
		avgN /= avgN_vec.size();
		avgGoodMatch /= avgGoodMatch_vec.size();

		TIME_avgGoodMatch /= TIME_avgGoodMatch_vec.size();
		TIME_avgDetect /= TIME_avgDetect_vec.size();
		TIME_avgCompute /= TIME_avgCompute_vec.size();

		cout << "TYPE_OF_FEATURE::[ " << typeOffeature << " ] " << endl;
		cout << "Average detected point: \t\t" << avgN << " " << endl;
		cout << "Average good Match: \t\t\t" << avgGoodMatch << " " << endl;
		cout << "\n" << endl;

		cout << "Average Detect keypoint (TIME): \t\t" << TIME_avgDetect << " (ms)\t" << 1000 / TIME_avgDetect << " (FPS)" << endl;
		cout << "Average Compute descrptor (TIME): \t\t" << TIME_avgCompute << " (ms)\t" << 1000 / TIME_avgCompute << " (FPS)" << endl;
		cout << "Average Detect & Compute (TIME): \t\t" << TIME_avgDetect + TIME_avgCompute << " (ms)\t" << 1000 / (TIME_avgDetect + TIME_avgCompute) << " (FPS)" << endl;
		cout << "Average Index the good Match (TIME): \t\t" << TIME_avgGoodMatch << " (ms)\t" << 1000 / TIME_avgGoodMatch << " (FPS)" << endl;
		cout << "\n" << endl;

		cout << "Average RATE: \t\t" << avgRate << " (%)" << endl;
		cout << "Average TIME: \t\t" << TIME_avgDetect + TIME_avgCompute + TIME_avgGoodMatch << " (ms)" << endl;
		cout << "Average FPS:  \t\t" << 1000 / (TIME_avgDetect + TIME_avgCompute + TIME_avgGoodMatch) << " (FPS)" << endl;
		cout << "[ ========== ====== ========== ]\n\n" << endl;
	}


}
