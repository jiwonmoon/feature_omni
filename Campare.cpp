#include "Compare.h"
#include "time.h"

#define DEBUGs

namespace F_test
{
	Compare::Compare() {}
	Compare::Compare(int _nfeatures, float _scaleFactor, int _nlevels,
		int _iniThFAST, int _minThFAST) {


		mpORBextractor = new ORBextractor(_nfeatures, _scaleFactor, _nlevels, _iniThFAST, _minThFAST);
		cout << endl << "=============================\nORB Extractor Parameters: " << endl;
		cout << "- Number of Features: " << _nfeatures << endl;
		cout << "- Scale Levels: " << _nlevels << endl;
		cout << "- Scale Factor: " << _scaleFactor << endl;
		cout << "- Initial Fast Threshold: " << _iniThFAST << endl;
		cout << "- Minimum Fast Threshold: " << _minThFAST << "\n=============================\n" << endl << endl;

	}



	void Compare::compare2img(const vector<Mat> img_set, const int idx1, const int idx2, const Mat mask, const Mat cube_mask, int typeOfFeature, int match_type) {
		cout << "[TwoViewTest >> compare2img (" << idx1 << " vs " << idx2 << ")]" << endl << endl;


		switch (typeOfFeature)
		{
		case 0:
		case 1:
		case 2:
			frame_1 = new Frame(img_set[idx1], mask, typeOfFeature);
			frame_2 = new Frame(img_set[idx2], mask, typeOfFeature);
			break;
		case 3:
			frame_1 = new Frame(img_set[idx1], mask, cube_mask, mpORBextractor);
			frame_2 = new Frame(img_set[idx2], mask, cube_mask, mpORBextractor);


			break;
		default:
			break;
		}

		int num_avg_Keypoint = (frame_1->N + frame_2->N) / 2;
		////draw left DC
		//Mat L_img = draw_info(frame_1);
		//Mat R_img = draw_info(frame_2);

		//imshow("L_img", L_img);	cv::moveWindow("L_img", 10, 50);
		//imshow("R_img", R_img);	cv::moveWindow("R_img", 10 + L_img.cols, 50);
		//waitKey(0);

		clock_t start_goodMatch = clock();
		vector<DMatch> goodMatches;
		switch (match_type)
		{
		case 1://brute force
			goodMatches = BF_find_goodMatches(frame_1->mDescriptors, frame_2->mDescriptors);
			break;

		case 2://knn search
			/// Find good Matches
			goodMatches = KNN_find_goodMatches(frame_1->mDescriptors, frame_2->mDescriptors);
			break;
		}
		double duration_goodMatch = (double)(clock() - start_goodMatch);

		int Avg_dist = 0, sum_dist = 0, n_dist = 0;
		for (int i = 0; i < goodMatches.size(); i++)
		{
			sum_dist += goodMatches[i].distance;
			n_dist++;
		}
		Avg_dist = sum_dist / n_dist;

		cout << "goodMatches size:: " << goodMatches.size() << endl;
		cout << "goodMatches Avg_dist:: " << Avg_dist << endl;


		float recog_Rate = (float)goodMatches.size() / (float)num_avg_Keypoint * 100;
		if (goodMatches.size() < 1)
		{
			cerr << "	!!!no goot Matcehs err: " << goodMatches.size() << endl << endl;
			exit(0);
		}

		avgN_vec.push_back(num_avg_Keypoint);
		avgGoodMatch_vec.push_back(goodMatches.size());
		avgRate_vec.push_back(recog_Rate);

		TIME_avgGoodMatch_vec.push_back(duration_goodMatch);

#ifdef DEBUG
		cout << "Detected keypoint size: (1):" << frame_1->N << " (2):" << frame_2->N << " =>	AVG: " << num_avg_Keypoint << endl;
		cout << "gootMatches,size(): " << goodMatches.size() << endl << endl;
		cout << "!!!Recongnize RATE: " << recog_Rate << " %" << endl << endl;
#endif




		///draw good_matches
		Mat imgMatches;
		drawMatches(frame_1->F_img, frame_1->mvKeys, frame_2->F_img, frame_2->mvKeys, goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		resize(imgMatches, imgMatches, Size(imgMatches.cols / 2, imgMatches.rows / 2));
		imshow("Good Matches", imgMatches);
		waitKey(0);
#ifdef DEBUG
		waitKey(0);
#endif
	}


	void Compare::translate_test(const vector<Mat> img_set, const Mat mask, const Mat cube_mask)
	{
		for (int idx = 0; idx < 6; idx++)
		{
			frame_1 = new Frame(img_set[idx], mask, cube_mask, mpORBextractor);

			Mat L_img = draw_info(frame_1);
			imshow("L_img", L_img);	cv::moveWindow("L_img", 10, 50);
			waitKey(0);
		}


	}


	vector<DMatch> Compare::BF_find_goodMatches(Mat desc1, Mat desc2)
	{
		vector<DMatch> matches;
		BFMatcher matcher(NORM_HAMMING);

		matcher.match(frame_1->mDescriptors, frame_2->mDescriptors, matches);
		cout << "matches.size(): " << matches.size() << endl;
		if (matches.size() < 4)
			exit(0);

		double minDist, maxDist;
		minDist = maxDist = matches[0].distance;
		for (int i = 1; i < matches.size(); i++)
		{
			double dist = matches[i].distance;
			if (dist < minDist) minDist = dist;
			if (dist > maxDist) maxDist = dist;
		}
		cout << "minDIst = " << minDist << endl;
		cout << "maxDist = " << maxDist << endl;

		vector<DMatch> goodMatches;
		double fTh = 4 * minDist;
		for (int i = 0; i < matches.size(); i++)
		{
			if (matches[i].distance <= max(fTh, 0.02))
				goodMatches.push_back(matches[i]);
		}

		return goodMatches;
	}


	vector<DMatch> Compare::KNN_find_goodMatches(Mat desc1, Mat desc2)
	{
		int k = 2;
		Mat indices;
		Mat dists;
		flann::Index flannIndex(desc2, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
		flannIndex.knnSearch(desc1, indices, dists, k, flann::SearchParams());

		vector<DMatch> goodMatches;
		float nndrRatio = 0.6f;
		for (int i = 0; i < desc1.rows; i++)
		{
			float d1, d2;
			d1 = (float)dists.at<int>(i, 0);
			d2 = (float)dists.at<int>(i, 1);

			if (indices.at<int>(i, 0) >= 0 && indices.at<int>(i, 1) >= 0 && d1 <= nndrRatio * d2)
			{
				DMatch match(i, indices.at<int>(i, 0), d1);
				goodMatches.push_back(match);
			}
		}
		return goodMatches;
	}


	Mat Compare::draw_info(const Frame* F)
	{
		Mat Lable_img(F->F_img.size(), CV_8U);
		drawKeypoints(F->F_img, F->mvKeys, Lable_img);

		KeyPoint element;
		for (int k = 0; k < F->mvKeys.size(); k++)
		{
			element = F->mvKeys[k];
			RotatedRect rRect = RotatedRect(element.pt, Size2f(element.size, element.size), element.angle);

			Point2f vertices[4];
			rRect.points(vertices);
			for (int i = 0; i < 4; i++)
				line(Lable_img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);

			circle(Lable_img, element.pt, cvRound(element.size / 2), Scalar(rand() % 256, rand() % 256, rand() % 256), 2);
		}

		resize(Lable_img, Lable_img, Size(Lable_img.cols, Lable_img.rows));
		return Lable_img;
	}


}