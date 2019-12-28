#ifndef COMPARE_H
#define COMPARE_H

#include "cv.hpp"
#include <iostream>
#include <time.h>

#include "Frame.h"
#include "ORBextractor.h"


extern vector<float> avgRate_vec;
extern vector<int> avgN_vec;
extern vector<int> avgGoodMatch_vec;

extern vector<double> TIME_avgGoodMatch_vec;
extern vector<double> TIME_avgDetect_vec;
extern vector<double> TIME_avgCompute_vec;

using namespace std;
using namespace cv;

namespace F_test
{
	class System;
	class Frame;	
	class ORBextractor;

	class Compare {

	public:
		Compare();
		Compare(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST);

		void compare2img(const vector<Mat> img_set, const int idx1, const int idx2, const Mat mask, const Mat cube_mask, int typeOfFeature, int match_tpye);
		void translate_test(const vector<Mat> img_set, const Mat mask, const Mat cube_mask);

		Frame* frame_1;
		Frame* frame_2;
		ORBextractor* mpORBextractor;

	protected:
		vector<DMatch> BF_find_goodMatches(Mat desc1, Mat desc2);
		vector<DMatch> KNN_find_goodMatches(Mat desc1, Mat desc2);
		Mat draw_info(const Frame* F);



	};
}



#endif