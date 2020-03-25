#include "cv.hpp"
#include <iostream>
#include <fstream>

#include "system.h"
#include "CamModelGeneral.h"
#include "ORBextractor.h"

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

class ORBextractor;

using namespace std;
using namespace cv;


vector<Mat> read_lafida_imgs(int start_flag, int end_img_cnt);
vector<Mat> make_x_simulation(Mat P_img, int level, int x_Dist, int width, int height);
Mat draw_info(Mat F, vector<KeyPoint> mvKeys);
Mat draw_patch(Mat& img, Point2f kpt);
void draw_patchs(Mat& img, vector<Point2f> kpts);


const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


/* main */
int main()
{
   int start_F_idx = 30, end_F_idx = 50;
   //cout << "SCAN START/END FRAME Idx: ";
   //cin >> start_F_idx;
   //cin >> end_F_idx;

   /* read data */
   vector<Mat> img_set = read_lafida_imgs(start_F_idx, end_F_idx); ///if 0 <- get all the every img set
   Mat mask = imread("/home/oooony/바탕화면/dataset/mask/lafida_mask.png", IMREAD_GRAYSCALE);
   Mat cube_mask = imread("/home/oooony/바탕화면/jiwon/CubemapSLAM/Masks/gray_lafida_cubemap_mask_650.png", IMREAD_GRAYSCALE);
   if (cube_mask.empty() || mask.empty())
   {
      std::cout << "fail to read the mask: " << std::endl;
      exit(0);
   }

   std::string settingFilePath("/home/oooony/바탕화면/jiwon/Config/lafida_cam0_params.yaml");


   //0: ORB, 1:BRISK, 2:AKAZE 3:ORB_EX
   F_test::System F_system(F_test::System::OMNI, F_test::System::ORB_EX, settingFilePath, img_set.size(), 50);
	std::vector<cv::Mat> LUT;
	cv::Mat LUT_x = F_system.mMap1;
	cv::Mat LUT_y = F_system.mMap2;
	cv::Mat LUT_x_inv = F_system.mMap1_inv;
	cv::Mat LUT_y_inv = F_system.mMap2_inv;
	LUT.push_back(LUT_x);
	LUT.push_back(LUT_y);
	LUT.push_back(LUT_x_inv);
	LUT.push_back(LUT_y_inv);


   int level = 150; int x_Dist = 1;
   //Mat P_origin = imread("../../graffiti/graff/img1.png", 0);
   Mat P_origin = imread("/home/oooony/바탕화면/dataset/graffiti/boat/img1.png", 0);
   Mat F_mask = imread("/home/oooony/바탕화면/dataset/graffiti/simulation_mask.png", 0);
   vector<Mat> simulation_X_P = make_x_simulation(P_origin, level, x_Dist, 650, 650);
   vector<Mat> simulation_X_F;

   for (int i = 0; i < simulation_X_P.size(); i++)
   {

      cv::Mat Fimg(img_set[0].rows, img_set[0].cols, CV_8U, cv::Scalar::all(0));
      F_system.CvtCubeMapToFisheye(simulation_X_P[i], Fimg);
      simulation_X_F.push_back(Fimg);

      //imshow("simulation_X_P", simulation_X_P[i]);
      //imshow("simulation_X_F", Fimg);
      //waitKey(0);
   }

   //imwrite("../../graffiti/graff/F.png", simulation_X_F[0]);


   /////////////////////
   int x_move = 100;
   int y_move = 130;
   Point2f center(simulation_X_F[0].cols /2, simulation_X_F[0].rows/2);
   Point2f LU(center.x - x_move-20, center.y + y_move);
   Point2f LD(center.x - x_move-20, center.y - y_move);
   Point2f RU(center.x + x_move+50, center.y + y_move);
   Point2f RD(center.x + x_move+50, center.y - y_move);

   int roi_rows = x_move * 2 + 70;
   int roi_cols = y_move * 2;

   Rect patch(LD.x, LD.y, roi_rows, roi_cols);
   Mat patch_mat = simulation_X_F[0](patch);


   vector<KeyPoint> mvKeys;
   Ptr<ORB> orbF = ORB::create(100);
   orbF->detect(patch_mat, mvKeys);
   int n_P = mvKeys.size();
   cout << n_P << endl;
   vector<vector<Point2f>> keypoints_P(level);
   vector<vector<Point2f>> keypoints(level);
   for (int i = 0; i < level; i++)
   {
      keypoints[i].resize(n_P);
      keypoints_P[i].resize(n_P);
   }

   for (int i = 0; i < mvKeys.size(); i++)
   {
      keypoints[0][i].x = mvKeys[i].pt.x + (LD.x);
      keypoints[0][i].y = mvKeys[i].pt.y + (LD.y);
   }

   /////////////////////



   int width = CamModelGeneral::GetCamera()->GetCubeFaceWidth(), height = CamModelGeneral::GetCamera()->GetCubeFaceHeight();

   for (int li = 0; li < level; li++)
   {
      for (int i = 0; i < n_P; i++)
      {
         if (li == 0)
         {
            float out_i, out_j;
            CamModelGeneral::GetCamera()->FisheyeToCubemap((keypoints[li][i].x), (keypoints[li][i].y), out_i, out_j);//fish2cube(fish -> cube)
            keypoints_P[li][i].x = floor(out_i) - width;
            keypoints_P[li][i].y = floor(out_j) - height;
         }
         else
         {
            keypoints_P[li][i].x = keypoints_P[li - 1][i].x - (x_Dist);
            keypoints_P[li][i].y = keypoints_P[li - 1][i].y;
         }

      }
   }

   for (int li = 0; li < level; li++)
   {
      for (int i = 0; i < n_P; i++)
      {

            double out_i, out_j;
            CamModelGeneral::GetCamera()->CubemapToFisheye(out_i, out_j, (keypoints_P[li][i].x + width), (keypoints_P[li][i].y + height));//fish2cube(fish -> cube)
            keypoints[li][i].x = out_i;
            keypoints[li][i].y = out_j;
      }
   }

   vector<Mat> F_clone(level), P_clone(level);
   for (int li = 0; li < level; li++)
   {
      simulation_X_F[li].copyTo(F_clone[li]);
      simulation_X_P[li].copyTo(P_clone[li]);
      F_clone.push_back(simulation_X_F[li]);
      P_clone.push_back(simulation_X_P[li]);


   }



   //vector<string> P_patch_dir;
   vector<string> P_dir;
   //vector<string> F_patch_dir;
   vector<string> F_dir;
   for (int n = 0; n < level; n++)
   {
      string file = to_string(n);
      string png = ".png";
      file.append(png);
   //   string patch_P_dir = "C:\\Users\\mjw31\\Desktop\\ex1_data\\pinhole\\patch\\";
   //   string patch_F_dir = "C:\\Users\\mjw31\\Desktop\\ex1_data\\distorted\\patch\\";      
      string img_P_dir = "C:\\Users\\mjw31\\Desktop\\ex1_data\\pinhole\\imgs\\";
      string img_F_dir = "C:\\Users\\mjw31\\Desktop\\ex1_data\\distorted\\imgs\\";
   //   patch_P_dir.append(file);
   //   patch_F_dir.append(file);
      img_P_dir.append(file);
      img_F_dir.append(file);

   //   P_patch_dir.push_back(patch_P_dir);
   //   F_patch_dir.push_back(patch_F_dir);
      P_dir.push_back(img_P_dir);
      F_dir.push_back(img_F_dir);

   }

   //for (int li = 0; li < level; li++)
   //{
   //   cv::resize(P_clone[li], P_clone[li], cv::Size(P_clone[li].cols * 1.25, P_clone[li].rows), 0, 0, CV_INTER_NN);
   //}
   for (int li = 0; li < level; li++)
   {

      draw_patchs(P_clone[li], keypoints_P[li]);
      draw_patchs(F_clone[li], keypoints[li]);
      imshow("P", P_clone[li]);
      imshow("F", F_clone[li]);
      waitKey(0);

      imwrite(P_dir[li], P_clone[li]);
      //imwrite(F_dir[li], F_clone[li]);
   }

   for (int n = 0; n < level; n++)
   {

      //Mat P_patch = draw_patch(P_clone[n], keypoints_P[n][0]);
      //Mat F_patch = draw_patch(F_clone[n], keypoints[n][0]);
      //for (int i = 0; i < n_P; i++)
      //{
      //   P_patch = draw_patch(P_clone[n], keypoints_P[n][0]);
      //   F_patch = draw_patch(F_clone[n], keypoints[n][0]);
      //}

      //imshow("P", P_clone[n]);
      //imshow("F", F_clone[n]);
      //imshow("P_patch", P_patch);
      //imshow("F_patch", F_patch);
      //waitKey(0);

      //imwrite(P_patch_dir[n], P_patch);
      //imwrite(F_patch_dir[n], F_patch);
      //imwrite(P_dir[n], P_clone[n]);
      //imwrite(F_dir[n], F_clone[n]);
   }

   F_system.TwoViewTest(img_set, mask, LUT, cube_mask, 1, 33); //(img term), (20)
   F_system.print_RESULT();

    //   F_system.direction_Test(simulation_X_F, simulation_X_P, keypoints, keypoints_P, F_mask, cube_mask, 33);
    //   waitKey(0);

   return 0;
}
/* main */
//Mat make_patch()
//{
//   Mat out;
//
//
//   return out;
//}
void draw_patchs(Mat& img, vector<Point2f> kpts)
{
   cvtColor(img, img, COLOR_GRAY2BGR);
   for (int i = 0; i < kpts.size(); i++)
   {
      Point2f LU(kpts[i].x - HALF_PATCH_SIZE, kpts[i].y + HALF_PATCH_SIZE);
      Point2f LD(kpts[i].x - HALF_PATCH_SIZE, kpts[i].y - HALF_PATCH_SIZE);
      Point2f RU(kpts[i].x + HALF_PATCH_SIZE, kpts[i].y + HALF_PATCH_SIZE);
      Point2f RD(kpts[i].x + HALF_PATCH_SIZE, kpts[i].y - HALF_PATCH_SIZE);
      line(img, LU, LD, Scalar(255, 0, 0), 4);
      line(img, LD, RD, Scalar(255, 0, 0), 4);
      line(img, RU, RD, Scalar(255, 0, 0), 4);
      line(img, LU, RU, Scalar(255, 0, 0), 4);
   }
}

Mat draw_patch(Mat& img, Point2f kpt)
{
   Mat out;
   cvtColor(img, img, COLOR_GRAY2BGR);

   Point2f LU(kpt.x - HALF_PATCH_SIZE, kpt.y + HALF_PATCH_SIZE);
   Point2f LD(kpt.x - HALF_PATCH_SIZE, kpt.y - HALF_PATCH_SIZE);
   Point2f RU(kpt.x + HALF_PATCH_SIZE, kpt.y + HALF_PATCH_SIZE);
   Point2f RD(kpt.x + HALF_PATCH_SIZE, kpt.y - HALF_PATCH_SIZE);
   line(img, LU, LD, Scalar(255, 0, 0), 4);
   line(img, LD, RD, Scalar(255, 0, 0), 4);
   line(img, RU, RD, Scalar(255, 0, 0), 4);
   line(img, LU, RU, Scalar(255, 0, 0), 4);
   //circle(img, kpt, 2, Scalar(0, 0, 255), 1);

   Rect patch(kpt.x - HALF_PATCH_SIZE+1, kpt.y - HALF_PATCH_SIZE+1, PATCH_SIZE, PATCH_SIZE);
   Mat patch_mat = img(patch);
   //imshow("patch", patch_mat);
   //waitKey(0);

   out = patch_mat;
   return out;
}


Mat draw_info(Mat F, vector<KeyPoint> mvKeys)
{
   Mat Lable_img(F.size(), CV_8U);
   drawKeypoints(F, mvKeys, Lable_img);

   KeyPoint element;
   for (int k = 0; k < mvKeys.size(); k++)
   {
      element = mvKeys[k];
      RotatedRect rRect = RotatedRect(element.pt, Size2f(element.size, element.size), element.angle);

      Point2f vertices[4];
      rRect.points(vertices);
      for (int i = 0; i < 4; i++)
         line(Lable_img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);

      circle(Lable_img, element.pt, cvRound(element.size / 2), Scalar(rand() % 256, rand() % 256, rand() % 256), 2);
   
      cout << element.pt << endl;
      cout << element.size << endl;
      cout << element.angle << endl << endl;

   }

   //resize(Lable_img, Lable_img, Size(Lable_img.cols, Lable_img.rows));
   return Lable_img;
}


vector<Mat> make_x_simulation(Mat P_img, int level, int x_Dist, int width, int height)
{
   resize(P_img, P_img, Size(width, height));
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


