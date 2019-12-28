#include "cv.hpp"
#include <iostream>
#include <fstream>

#include "system.h"
#include "CamModelGeneral.h"
#include "ORBextractor.h"

//Extern Global Parameter for Print Result
vector<float> avgRate_vec;
vector<int> avgN_vec;
vector<int> avgGoodMatch_vec;

vector<double> TIME_avgGoodMatch_vec;
vector<double> TIME_avgDetect_vec;
vector<double> TIME_avgCompute_vec;


using namespace std;
using namespace cv;

class ORBextractor;


vector<Mat> read_lafida_imgs(int start_flag, int end_img_cnt);
vector<Mat> make_x_simulation(Mat P_img);


/* main */
int main()
{
	int start_F_idx = 30, end_F_idx = 38;
	//cout << "SCAN START/END FRAME Idx: ";
	//cin >> start_F_idx;
	//cin >> end_F_idx;

	/* read data */
	vector<Mat> img_set = read_lafida_imgs(start_F_idx, end_F_idx);	///if 0 <- get all the every img set
	Mat mask = imread("../../dataset/mask/lafida_mask.png", IMREAD_GRAYSCALE);
	Mat cube_mask = imread("../../dataset/cube_mask/gray_lafida_cubemap_mask_450.png", IMREAD_GRAYSCALE);
	if (cube_mask.empty() || mask.empty())
	{
		std::cout << "fail to read the mask: " << std::endl;
		exit(0);
	}

	std::string settingFilePath("../../Config/lafida_cam0_params.yaml");


	//0: ORB, 1:BRISK, 2:AKAZE 3:ORB_EX
	F_test::System F_system(F_test::System::OMNI, F_test::System::ORB_EX, settingFilePath, img_set.size(), 50);


	///make cube img
	int offset = 0;
	int width = CamModelGeneral::GetCamera()->GetCubeFaceWidth(), height = CamModelGeneral::GetCamera()->GetCubeFaceHeight();
	cv::Mat cubemapImg(height * 3, width * 3, CV_8U, cv::Scalar::all(0));
	cv::Mat cubemapImg_front = cubemapImg.rowRange(height, 2 * height).colRange(width, 2 * width);
	cv::Mat cubemapImg_left = cubemapImg.rowRange(height, 2 * height).colRange(0 + offset, width + offset);
	cv::Mat cubemapImg_right = cubemapImg.rowRange(height, 2 * height).colRange(2 * width - offset, 3 * width - offset);
	cv::Mat cubemapImg_upper = cubemapImg.rowRange(0 + offset, height + offset).colRange(width, 2 * width);
	cv::Mat cubemapImg_lower = cubemapImg.rowRange(2 * height - offset, 3 * height - offset).colRange(width, 2 * width);
	F_system.CvtFisheyeToCubeMap_reverseQuery_withInterpolation(cubemapImg, img_set[0], cv::INTER_LINEAR);


	Mat P_origin = imread("../../graffiti/boat/img1.png", 0);
	Mat P_mask = imread("../../graffiti/simulation_mask.png", 0);
	vector<Mat> simulation_X = make_x_simulation(P_origin);
	vector<Mat> simulation_X_F;


	
	for (int i = 0; i < simulation_X.size(); i++)
	{

		cv::Mat Fimg(img_set[0].rows, img_set[0].cols, CV_8U, cv::Scalar::all(0));
		F_system.CvtCubeMapToFisheye(simulation_X[i], Fimg);
		simulation_X_F.push_back(Fimg);

		//imshow("simulation_X", simulation_X[i]);
		//imshow("simulation_X_P", Fimg);
		//waitKey(0);
	}

	imwrite("../../graffiti/graff/F.png", simulation_X_F[0]);

	F_system.direction_Test(simulation_X_F, P_mask, cube_mask, 33);



	waitKey(0);

	return 0;
}
/* main */


vector<Mat> make_x_simulation(Mat P_img)
{
	resize(P_img, P_img, Size(480, 480));
	int x_Dist = 30;
	int level = 6;
	vector<Mat> x_P_img;
	x_P_img.reserve(level);

	for (int i = 0; i < level; i++)
	{
		Mat temp(P_img.rows, P_img.cols, CV_8U, cv::Scalar::all(0));
		for (int iw = 0; iw < P_img.cols; iw++)
		{
			for (int ih = 0; ih < P_img.rows; ih++)
			{
				uchar intensity = P_img.at<uchar>(ih, iw);

				if (iw - x_Dist * i < 0)
					continue;
				temp.at<uchar>(ih, iw - x_Dist * i) = intensity;

			}
		}

		x_P_img.push_back(temp);
	}

	return x_P_img;
}

vector<Mat> read_lafida_imgs(int start_flag, int end_img_cnt)
{
	std::string fisheyeImgPath = "../../dataset/outdoor_rotation/imgs/cam0/";
	std::ifstream fin("../../dataset/outdoor_rotation/images_and_timestamps.txt");

	// Retrieve paths to images
	std::vector<std::string> fisheyeImgNames;
	std::vector<double> vTimestamps;
	std::string line;

	while (std::getline(fin, line))
	{
		std::stringstream ss(line);
		double ts;
		ss >> ts;
		std::string name;
		ss >> name;
		size_t p = name.find_last_of("/");
		name = name.substr(p + 1, name.length());

		vTimestamps.push_back(ts);
		fisheyeImgNames.push_back(name);
	}


	// Main loop
	vector<Mat> img_set;
	cv::Mat im;
	int ni = 0;
	int end_flag = 0;

	if (end_img_cnt == 0)
		end_flag = fisheyeImgNames.size();
	else
		end_flag = end_img_cnt;

	ni = start_flag;
	while (true)
	{
		if (ni == end_flag)
			break;

		std::string fisheyeImgName = fisheyeImgPath + fisheyeImgNames[ni];
		im = cv::imread(fisheyeImgName, 0);
		img_set.push_back(im);

		if (im.empty())
		{
			cerr << endl
				<< "Failed to read camera" << endl;
			break;
		}
		if (ni == fisheyeImgNames.size())
		{
			cout << "frame end" << endl;
			break;
		}
		ni++;
	}

	const int imageNameCnt = fisheyeImgNames.size();
	std::cout << "ImageNameCnt find " << imageNameCnt << " image names" << std::endl;
	const int imageCnt = img_set.size();
	std::cout << "ImageCnt find " << imageCnt << " images" << std::endl;
	std::cout << "START/END FRAME Idx: [" << start_flag << " , " << end_img_cnt << "]" << std::endl;


	return img_set;
}



