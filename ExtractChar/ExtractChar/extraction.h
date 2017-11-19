/* --------------------------------------------------------
* author£ºlivezingy
*
* BLOG£ºhttp://www.livezingy.com
*
* Development Environment£º
*      Visual Studio V2013
*      opencv3.1
*
* Version£º
*      V1.0    20171119
--------------------------------------------------------- */
#ifndef EXTRACTION_H
#define EXTRACTION_H

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;
namespace extraction {

	struct SWTPoint2d {
		int x;
		int y;
		float SWT;
	};

	typedef std::pair<SWTPoint2d, SWTPoint2d> SWTPointPair2d;
	typedef std::pair<cv::Point, cv::Point>   SWTPointPair2i;

	struct Point2dFloat {
		float x;
		float y;
	};

	struct Ray {
		SWTPoint2d p;
		SWTPoint2d q;
		std::vector<SWTPoint2d> points;
	};

	struct Point3dFloat {
		float x;
		float y;
		float z;
	};


	struct Chain {
		int p;
		int q;
		float dist;
		bool merged;
		Point2dFloat direction;
		std::vector<int> components;
	};

	

	bool Point2dSort(SWTPoint2d const & lhs,SWTPoint2d const & rhs);

	void textDetection(const cv::Mat& input, bool dark_on_light);

	void testConnected(Mat& img);

	void getAdjacentComponent(Mat& SWTimg, Mat& comLabels, Mat& stats, Mat& centroids, std::vector<Vec3b>& colors, int nLabels);

	void getSingleComponent(Mat& SWTimg, Mat& comLabels, Mat& outPut, int index);

	void SWTMedianFilter(Mat& SWTImage, std::vector<Ray> & rays);

	void strokeWidthTransform(const Mat& edgeImage,
		Mat& gradientX,
		Mat& gradientY,
		bool dark_on_light,
		Mat& SWTImage,
		std::vector<Ray> & rays);

	void strokeWidthTransform(const cv::Mat& edgeImage,
		cv::Mat& gradientX,
		cv::Mat& gradientY,
		bool dark_on_light,
		cv::Mat& SWTImage,
		std::vector<Ray> & rays);

	void SWTMedianFilter(cv::Mat& SWTImage, std::vector<Ray> & rays);

	void histFilter(Mat& img, Mat& labels, Mat& stats, int nLabels);

	void writeToExcel(Mat outputImage, string fileName);



	//std::vector< std::vector<SWTPoint2d> > findLegallyConnectedComponents(const cv::Mat& input,cv::Mat& SWTImage, std::vector<Ray> & rays);
	cv::Mat findLegallyConnectedComponents(const cv::Mat& input, cv::Mat& SWTImage, std::vector<Ray> & rays);
	std::vector< std::vector<SWTPoint2d> >
		findLegallyConnectedComponentsRAY(IplImage * SWTImage,
		std::vector<Ray> & rays);

	void componentStats(IplImage * SWTImage,
		const std::vector<SWTPoint2d> & component,
		float & mean, float & variance, float & median,
		int & minx, int & miny, int & maxx, int & maxy);

	void filterComponents(cv::Mat& SWTImage,
		std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<std::vector<SWTPoint2d> > & validComponents,
		std::vector<Point2dFloat> & compCenters,
		std::vector<float> & compMedians,
		std::vector<SWTPoint2d> & compDimensions,
		std::vector<SWTPointPair2d > & compBB);

	std::vector<Chain> makeChains(const cv::Mat& colorImage,
		std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<Point2dFloat> & compCenters,
		std::vector<float> & compMedians,
		std::vector<SWTPoint2d> & compDimensions,
		std::vector<SWTPointPair2d > & compBB);

}

#endif // EXTRACTION_H

