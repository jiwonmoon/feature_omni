#include "cv.hpp"
#include <iostream>
#include <fstream>

#include "system.h"
#include "CamModelGeneral.h"

//Extern Global Parameter for Print Result
vector<float> avgRate_vec;
vector<int> avgN_vec;
vector<int> N1_vec;
vector<int> N2_vec;
vector<int> avgGoodMatch_vec;
vector<int> HammingDistance;

vector<double> TIME_DC1_vec;
vector<double> TIME_DC2_vec;
vector<double> TIME_avgGoodMatch_vec;
vector<double> TIME_avgDetect_vec;
vector<double> TIME_avgCompute_vec;


using namespace std;
using namespace cv;


vector<Mat> read_lafida_imgs(int start_flag, int end_img_cnt);


/* main */
int main()
{
	int start_F_idx = 30, end_F_idx = 50;
	//cout << "SCAN START/END FRAME Idx: ";
	//cin >> start_F_idx;
	//cin >> end_F_idx;
	
	/* read data */
	vector<Mat> img_set = read_lafida_imgs(start_F_idx, end_F_idx);	///if 0 <- get all the every img set
	Mat mask = imread("/home/oooony/바탕화면/dataset/mask/lafida_mask.png", IMREAD_GRAYSCALE);
	Mat cube_mask = imread("/home/oooony/바탕화면/jiwon/CubemapSLAM/Masks/gray_lafida_cubemap_mask_650.png", IMREAD_GRAYSCALE);
	if (cube_mask.empty() || mask.empty())
	{
		std::cout << "fail to read the mask: " << std::endl;
		exit(0);
	}

	std::string settingFilePath("/home/oooony/바탕화면/jiwon/Config/lafida_cam0_params.yaml");


	//0: ORB, 1:BRISK, 2:AKAZE 3:ORB_EX
	F_test::System F_system(F_test::System::OMNI, F_test::System::ORB_EX, settingFilePath,img_set.size(), 500);

	std::vector<cv::Mat> LUT;
	cv::Mat LUT_x = F_system.mMap1;
	cv::Mat LUT_y = F_system.mMap2;
	cv::Mat LUT_x_inv = F_system.mMap1_inv;
	cv::Mat LUT_y_inv = F_system.mMap2_inv;
	LUT.push_back(LUT_x);
	LUT.push_back(LUT_y);
	LUT.push_back(LUT_x_inv);
	LUT.push_back(LUT_y_inv);

	// int rows = F_system.mMap1.rows;
	// int cols = F_system.mMap1.cols;
	// for(int i =0 ; i < rows ; i++)
	// {
	// 	for(int j=0; j < cols ; j++)
	// 	{
	// 		cout <<"fish: " << i << " , " << j << "cube: " <<  F_system.mMap1.at<float>(i, j) << " , " << F_system.mMap2.at<float>(i, j) << endl;
	// 	}
	// }



	F_system.TwoViewTest(img_set, mask, LUT, cube_mask, 0, 33);//(img term), (20)
	F_system.print_RESULT();




	waitKey(0);

	return 0;
}
/* main */


//outdoor_rotation
//outdoor_static
//indoor_static
vector<Mat> read_lafida_imgs(int start_flag, int end_img_cnt)
{
	std::string fisheyeImgPath = "/home/oooony/바탕화면/dataset/outdoor_static/imgs/cam0/";
	std::ifstream fin("/home/oooony/바탕화면/dataset/outdoor_static/images_and_timestamps.txt");

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



