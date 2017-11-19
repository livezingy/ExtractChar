/* --------------------------------------------------------
* author：livezingy
*
* BLOG：http://www.livezingy.com
*
* Development Environment：
*      Visual Studio V2013
*      opencv3.1
*
* Version：
*      V1.0    20171119
--------------------------------------------------------- */

#include "stdafx.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <time.h>
#include <utility>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <fstream>

using namespace cv;
using namespace std;

#include "extraction.h"

Mat sourceImg;

#define PI 3.14159265
#define MAX_SWT 60;
#define MIN_SWT 2;

namespace extraction 
{

	void textDetection(const Mat& input, bool dark_on_light)
	{
		assert(input.depth() == CV_8U);
		assert(input.channels() == 3);
		input.copyTo(sourceImg);

		std::cout << "Running textDetection with dark_on_light " << dark_on_light << std::endl;

		// Convert to grayscale
		Mat grayImage(input.size(), CV_8UC1);
		cvtColor(input, grayImage, CV_RGB2GRAY);
		// Create Canny Image
		double threshold_low = 175;
		double threshold_high = 320;
		Mat edgeImage(input.size(), CV_8UC1);
		Canny(grayImage, edgeImage, threshold_low, threshold_high, 3);

		imshow("canny", edgeImage);
		//imwrite("canny.png",edgeImage);

		// Create gradient X, gradient Y
		Mat gaussianImage(input.size(), CV_32FC1);
		grayImage.convertTo(gaussianImage, CV_32FC1, 1. / 255.);
		GaussianBlur(gaussianImage, gaussianImage, Size(5, 5), 0);
		Mat gradientX(input.size(), CV_32FC1);
		Mat gradientY(input.size(), CV_32FC1);
		Scharr(gaussianImage, gradientX, -1, 1, 0);
		Scharr(gaussianImage, gradientY, -1, 0, 1);

		GaussianBlur(gradientX, gradientX, Size(3, 3), 0);
		GaussianBlur(gradientY, gradientY, Size(3, 3), 0);


		// Calculate SWT and return ray vectors
		std::vector<Ray> rays;
		Mat SWTImage(input.size(), CV_32FC1);
		for (int row = 0; row < input.rows; row++)
		{
			float* ptr = (float*)SWTImage.ptr(row);

			for (int col = 0; col < input.cols; col++)
			{
				*ptr++ = -1;
			}
		}
		strokeWidthTransform(edgeImage, gradientX, gradientY, dark_on_light, SWTImage, rays);

		SWTMedianFilter(SWTImage, rays);

		Mat saveSWT(input.size(), CV_8UC1);
		SWTImage.convertTo(saveSWT, CV_8UC1, 255);

		Mat image_open(input.size(), CV_32FC1);

		Mat element = getStructuringElement(MORPH_CROSS, Size(4, 4));//MORPH_CROSS MORPH_RECT MORPH_ELLIPSE 
		morphologyEx(SWTImage, image_open, MORPH_OPEN, element);
		imshow("开运算1", image_open);
		//imwrite("image_open11.png", image_open);

		testConnected(image_open);

		imshow("SWTImage1", saveSWT);
		//imwrite("SWTImage1.png", saveSWT);
		

		cvWaitKey();
	}

	void testConnected(Mat& img)
	{

		Mat  img_color, stats, centroids;
		Mat labels(img.size(), CV_32S);

		Mat img_edge = img > 0;

		Mat copyImage(img.size(), CV_32FC1);
		img.copyTo(copyImage);

		int i = 0;

		int nLabels = cv::connectedComponentsWithStats(img_edge, labels, stats, centroids, 4);

		// Show connected components with random colors
		std::vector<Vec3b> colors(nLabels);
		colors[0] = Vec3b(0, 0, 0);//background

		//1.连通域像素总数小于minArea 2.连通域长或宽大于maxSize 3. 连通域长或宽小于minSize
		//满足上述三点条件的认为其为无效连通域
		int maxSize = 90;
		int minSize = 15;
		int minArea = 40;
		int maxArea = 1000;


		for (int label = 1; label < nLabels; ++label)
		{
			colors[label] = Vec3b((rand() & 200), (rand() & 200), (rand() & 200));

			
			if ((stats.at<int>(label, cv::CC_STAT_AREA) < minArea) || (stats.at<int>(label, cv::CC_STAT_WIDTH) > maxSize) || (stats.at<int>(label, cv::CC_STAT_HEIGHT) > maxSize)
				|| (stats.at<int>(label, cv::CC_STAT_WIDTH) < minSize) || (stats.at<int>(label, cv::CC_STAT_HEIGHT) < minSize))
			{
				colors[label] = Vec3b(0, 0, 0); // small regions are painted with black too.
			}
			
			//centroids.at<double>(i, 0),			centroids.at<double>(i, 1)
		}
	
		
		//计算每个连通域笔画宽度的直方图，淘汰笔画宽度分布不合理的连通域
		Mat histSWT(nLabels, 16, CV_16U);
		histSWT = Mat::zeros(histSWT.size(), CV_16U);
		for (int r = 0; r < img.rows; ++r)
		{
			float* ptr = (float*)img.ptr(r);

			for (int c = 0; c < img.cols; ++c)
			{
				int tmpRow = labels.at<int>(r, c);
				int tmpCol = round(*ptr++);//对笔画数四舍五入后作为列数

				if (tmpCol != -1)
				{
					histSWT.at<unsigned short>(tmpRow, tmpCol) = histSWT.at<unsigned short>(tmpRow, tmpCol) + 1;
				}
			}
		}

		for (i = 1; i < nLabels; i++)
		{
			if (colors[i] != Vec3b(0, 0, 0))
			{
				int invalidSummin = histSWT.at<unsigned short>(i, 1) + histSWT.at<unsigned short>(i, 2);

				int invalidSummax = histSWT.at<unsigned short>(i, 10)+histSWT.at<unsigned short>(i, 11) + histSWT.at<unsigned short>(i, 12) + histSWT.at<unsigned short>(i, 13) + histSWT.at<unsigned short>(i, 14);

				float ratioMin = (float)invalidSummin / stats.at<int>(i, CC_STAT_AREA);

				float ratioMax = (float)invalidSummax / stats.at<int>(i, CC_STAT_AREA);

				if ((ratioMin > 0.5) || (ratioMax > 0.5))
				{
					colors[i] = Vec3b(0, 0, 0);
				}
			}
		}


		//1. 将连通域为黑色的像素的笔画宽度值赋值为-1
		//2. 以连通域的左上角为起点，长宽较大值为边长的正方形区域内有效像素面积小于maxArea的认定其为无效连通域
		//3. 该操作暂定循环执行两次，后续看状况可更正其执行次数
		for (int t = 0; t < 2; t++)
		{
			for (int r = 0; r < copyImage.rows; ++r)
			{
				float* ptr = (float*)copyImage.ptr(r);

				for (int c = 0; c < copyImage.cols; ++c)
				{
					int la = labels.at<int>(r, c);

					if (colors[la] == Vec3b(0, 0, 0))
					{
						*ptr = -1;
					}

					*ptr++;
				}
			}

			for (i = 1; i < nLabels; i++)
			{
				int sum = 0;
				if (colors[i] != Vec3b(0, 0, 0))
				{
					int left = stats.at<int>(i, cv::CC_STAT_LEFT);
					int top = stats.at<int>(i, cv::CC_STAT_TOP);

					int length = max(stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
					int width = min(length + left, copyImage.cols);
					int height = min(length + top, copyImage.rows);


					for (int r = top; r < height; ++r)
					{
						for (int c = left; c < width; ++c)
						{
							if (copyImage.at<int>(r, c) != -1)
							{
								sum++;
							}
						}
					}

					if (sum < maxArea)
					{
						colors[i] = Vec3b(0, 0, 0);
					}
				}
			}

			for (int r = 0; r < copyImage.rows; ++r)
			{
				float* ptr = (float*)copyImage.ptr(r);

				for (int c = 0; c < copyImage.cols; ++c)
				{
					int la = labels.at<int>(r, c);

					if (colors[la] == Vec3b(0, 0, 0))
					{
						*ptr = -1;
					}

					*ptr++;
				}
			}
		}

		Mat saveSWT(copyImage.size(), CV_8UC1);
		copyImage.convertTo(saveSWT, CV_8UC1, 255);

		Mat image_open(saveSWT.size(), CV_32FC1);//, image_close, image_tophat, image_blackhat, image_gredient;

		//GaussianBlur(saveSWT, saveSWT, Size(11, 11), 0);
		Mat element = getStructuringElement(MORPH_CROSS, Size(4, 4));//MORPH_CROSS MORPH_RECT MORPH_ELLIPSE 
		morphologyEx(saveSWT, image_open, MORPH_OPEN, element);
		imshow("开运算", image_open);
		//imwrite("image_open2.png", image_open);

		imshow("SWTImage", saveSWT);
		//imwrite("saveSWT2.png", saveSWT);

		Mat dst(img.size(), CV_8UC3);
		for (int r = 0; r < dst.rows; ++r)
		{
			for (int c = 0; c < dst.cols; ++c)
			{
				int label = labels.at<int>(r, c);
				Vec3b &pixel = dst.at<Vec3b>(r, c);
				pixel = colors[label];
			}
		}
		
		// Text labels with area of each cc (except background)
     
		for (i = 1; i< nLabels; i++)
		{
			if (colors[i] != Vec3b(0, 0, 0))
			{
				//float a = stats.at<int>(i, CC_STAT_AREA);
				Point org(centroids.at<double>(i, 0),
					centroids.at<double>(i, 1));
				String txtarea;
				std::ostringstream buff;
				buff << i;
				txtarea = buff.str();
				//putText(dst, txtarea, org, 1, 1, Scalar(255, 255, 255), 1);

				cout << i << "\t" << centroids.at<double>(i, 0) << "\t" << centroids.at<double>(i, 1) << "\t" << stats.at<int>(i, cv::CC_STAT_WIDTH) << "\t" << stats.at<int>(i, cv::CC_STAT_HEIGHT) << "\t" << stats.at<int>(i, cv::CC_STAT_LEFT) << "\t" << stats.at<int>(i, cv::CC_STAT_TOP) << std::endl;
			}
		}

		cv::imshow("Connected Components", dst);

		getAdjacentComponent(dst, labels, stats, centroids, colors, nLabels);
		
	}

	//较接近连通域区域合并，取出原图中合并后的连通域单独进行二值化，二值化后的区域逐一与文字进行比较匹配
	//较接近连通域区域合并，取出原图中合并后的连通域单独进行二值化，二值化后的区域逐一与文字进行比较匹配
	void getAdjacentComponent(Mat& SWTimg, Mat& comLabels, Mat& stats, Mat& centroids, std::vector<Vec3b>& colors, int nLabels)
	{
		int value_1 = 45;
		float gain = 1.2;//将找到的矩形放大1.2倍
		Mat bComponent = Mat::zeros(comLabels.size(), CV_8U);
		int centoidA_0, centoidA_1, centoidB_0, centoidB_1;

		std::vector<Rect> rect;
		int nonZeroNum = 0;

		for (int i = 1; i < nLabels - 1; i++)
		{
			if (colors[i] == Vec3b(0, 0, 0))
			{
				bComponent.at<uchar>(i) = 1;
			}
			else
			{
				nonZeroNum++;
			}
		}

		//a.外接矩形区域有重合  b.外接矩形区域垂直距离小于设定值  c.外接矩形区域水平距离小于设定值
		//上述三种状况合并为一个区域
		//经过确认，连通域的排序并未根据质心从上至下或从左至右排列，因此在判断区域时，需要遍历每一个连通域
		for (int i = 1; i < nLabels - 1; i++)
		{
			if (0 == bComponent.at<uchar>(i))
			{
				centoidA_0 = centroids.at<double>(i, 0);
				centoidA_1 = centroids.at<double>(i, 1);
				bComponent.at<uchar>(i) = 1;

				nonZeroNum++;

				Rect tmpRect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));

				for (int j = i + 1; j < nLabels; j++)
				{
					if (0 == bComponent.at<uchar>(j))
					{
						centoidB_0 = centroids.at<double>(j, 0);
						centoidB_1 = centroids.at<double>(j, 1);

						nonZeroNum++;

						if ((abs(centoidA_0 - centoidB_0) < value_1) && (abs(centoidA_1 - centoidB_1) < value_1))
						{
							int tmpLeft = stats.at<int>(j, cv::CC_STAT_LEFT);
							int tmpTop = stats.at<int>(j, cv::CC_STAT_TOP);
							int tmpWidth = stats.at<int>(j, cv::CC_STAT_WIDTH);
							int tmpHeight = stats.at<int>(j, cv::CC_STAT_HEIGHT);

							//Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);

							bComponent.at<uchar>(j) = 1;

							//更新当前合并后连通域的左点，宽
							if (tmpRect.x > tmpLeft)
							{
								if ((tmpRect.x + tmpRect.width) < (tmpLeft + tmpWidth))
								{
									tmpRect.width = tmpWidth;
								}
								else
								{
									tmpRect.width = tmpRect.x + tmpRect.width - tmpLeft;
								}

								tmpRect.x = tmpLeft;
							}
							else
							{
								if ((tmpRect.x + tmpRect.width) < (tmpLeft + tmpWidth))
								{
									tmpRect.width = tmpLeft + tmpWidth - tmpRect.x;
								}
							}

							//更新当前合并后连通域的上点，高
							if (tmpRect.y > tmpTop)
							{
								if ((tmpRect.y + tmpRect.height) < (tmpTop + tmpHeight))
								{
									tmpRect.height = tmpHeight;
								}
								else
								{
									tmpRect.height = tmpRect.y + tmpRect.height - tmpTop;
								}

								tmpRect.y = tmpTop;
							}
							else
							{
								if ((tmpRect.y + tmpRect.height) < (tmpTop + tmpHeight))
								{
									tmpRect.height = tmpTop + tmpHeight - tmpRect.y;
								}
							}
						}
					}
				}
				if (tmpRect.width > tmpRect.height)
				{
					tmpRect.height = tmpRect.width;		
				}
				else
				{
					tmpRect.width = tmpRect.height;
				}


				int tmpGain = round(gain * tmpRect.height);
				int tmpOffset = (tmpGain - tmpRect.height) / 2;

				if (tmpRect.x > tmpOffset)
				{
					tmpRect.x = tmpRect.x - tmpOffset;

					tmpRect.width = tmpGain;
				}
				else
				{
					tmpRect.x = 0;

					tmpRect.width = tmpGain + tmpOffset - tmpRect.x;
				}

				if (tmpRect.y > tmpOffset)
				{
					tmpRect.y = tmpRect.y - tmpOffset;

					tmpRect.height = tmpGain;
				}
				else
				{
					tmpRect.y = 0;
					tmpRect.height = tmpGain + tmpOffset - tmpRect.y;
				}

				if ((tmpRect.x + tmpRect.width) > SWTimg.cols)
				{
					tmpRect.width = SWTimg.cols - tmpRect.x;
				}
				if ((tmpRect.y + tmpRect.height) > SWTimg.rows)
				{
					tmpRect.height = SWTimg.rows - tmpRect.y;
				}

				rect.push_back(tmpRect);
			}
		}
		
		for (int r = 0; r < rect.size(); r ++)
		{
			Rect tmpRect = rect[r];
			rectangle(sourceImg, tmpRect.tl(), tmpRect.br(), Scalar(255, 0, 0), 3, 8, 0);
			Mat out = sourceImg(tmpRect);
			// Convert to grayscale
			Mat grayImage(out.size(), CV_8UC1);
			cvtColor(out, grayImage, CV_RGB2GRAY);
			Mat dst = Mat::zeros(out.size(), CV_8UC1);
			//threshold(grayImage, dst, 0, 255, THRESH_BINARY);
			
			adaptiveThreshold(grayImage, dst, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 20);
			/*
			switch (r)
			{
			case 0:
				imshow("out0", dst);
				//imwrite("out0.png", dst);
				break;
				
			case 1:
				imshow("out1", dst);
				//imwrite("out1.png", dst);
				break;

			case 2:
				imshow("out2", dst);
				//imwrite("out2.png", dst);
				break;

			case 3:
				imshow("out3", dst);
				//imwrite("out3.png", dst);
				break;

			case 4:
				imshow("out4", dst);
				//imwrite("out4.png", dst);
				break;

			case 11:
				imshow("out11", dst);
				break;

			case 12:
				imshow("out12", dst);
				break;

			case 13:
				imshow("out13", dst);
				break;

			case 10:
				imshow("out10", dst);
				break;
				

			default:
				break;

			}
			*/

		}

		cv::imshow("rectangle", sourceImg);
		imwrite("sourceImg32.png", sourceImg);
	
	}


	//单独提取单个连通域，返回矩阵前景为黑色，背景为白色
	void getSingleComponent(Mat& SWTimg, Mat& comLabels, Mat& outPut, int index)
	{
		outPut = Mat::zeros(SWTimg.size(), CV_8UC1);
		for (int r = 0; r < SWTimg.rows; ++r)
		{
			uchar* ptr = (uchar*)outPut.ptr(r);
			for (int c = 0; c < SWTimg.cols; ++c)
			{
				if (index == comLabels.at<int>(r, c))
				{
					if ((uchar)SWTimg.at<float>(r, c) == 5 || (uchar)SWTimg.at<float>(r, c) == 6)
					{
						ptr[c] = (uchar)SWTimg.at<float>(r, c);
					}
					else
					{
						ptr[c] = 255;
					}

					//outPut.at<int>(r, c) = 0;
				}
				else
				{
					ptr[c] = 255;
					//outPut.at<int>(r, c) = 1;
				}
			}
		}
		//imwrite("outPut.png", outPut);
		imshow("outPut1111", outPut);

		//writeToExcel(outPut, "2-44.xls");

		//cv::waitKey();
	}
	bool Point2dSort(const SWTPoint2d &lhs, const SWTPoint2d &rhs) 
	{
		return lhs.SWT < rhs.SWT;
	}

	void SWTMedianFilter(Mat& SWTImage, std::vector<Ray> & rays)
	{
		for (auto& rit : rays)
		{
			for (auto& pit : rit.points)
			{
				pit.SWT = SWTImage.at<float>(pit.y, pit.x);
			}
			std::sort(rit.points.begin(), rit.points.end(), &Point2dSort);
			float median = (rit.points[rit.points.size() / 2]).SWT;

			for (auto& pit : rit.points)
			{
				float tmpVal = std::min(pit.SWT, median);

				if (tmpVal < 15)
				{
					SWTImage.at<float>(pit.y, pit.x) = tmpVal;
				}
				else
				{
					SWTImage.at<float>(pit.y, pit.x) = -1;
				}

			}
		}
	}

	void strokeWidthTransform(const Mat& edgeImage,
		Mat& gradientX,
		Mat& gradientY,
		bool dark_on_light,
		Mat& SWTImage,
		std::vector<Ray> & rays) {
		// First pass
		float prec = .05;
		for (int row = 0; row < edgeImage.rows; row++){
			const uchar* ptr = (const uchar*)edgeImage.ptr(row);
			for (int col = 0; col < edgeImage.cols; col++){
				if (*ptr > 0) {
					/*
					struct Ray {
					SWTPoint2d p;
					SWTPoint2d q;
					std::vector<SWTPoint2d> points;
					};
					*/
					Ray r;

					/*
					struct SWTPoint2d {
					int x;
					int y;
					float SWT;
					};
					*/
					SWTPoint2d p;
					p.x = col;
					p.y = row;
					r.p = p;
					std::vector<SWTPoint2d> points;
					points.push_back(p);

					float curX = (float)col + 0.5;
					float curY = (float)row + 0.5;
					int curPixX = col;
					int curPixY = row;
					float G_x = gradientX.at<float>(row, col);
					float G_y = gradientY.at<float>(row, col);
					// normalize gradient
					float mag = sqrt((G_x * G_x) + (G_y * G_y));
					if (dark_on_light){
						G_x = -G_x / mag;
						G_y = -G_y / mag;
					}
					else {
						G_x = G_x / mag;
						G_y = G_y / mag;

					}
					while (true) {
						//float prec = .05;
						//float curX = (float)col + 0.5;
						//float curY = (float)row + 0.5;
						//int curPixX = col;
						//int curPixY = row;
						curX += G_x*prec;
						curY += G_y*prec;
						if ((int)(floor(curX)) != curPixX || (int)(floor(curY)) != curPixY) {
							curPixX = (int)(floor(curX));
							curPixY = (int)(floor(curY));
							// check if pixel is outside boundary of image
							if (curPixX < 0 || (curPixX >= SWTImage.cols) || curPixY < 0 || (curPixY >= SWTImage.rows)) {
								break;
							}
							SWTPoint2d pnew;
							pnew.x = curPixX;
							pnew.y = curPixY;
							points.push_back(pnew);

							uchar edgeVal = edgeImage.at<uchar>(curPixY, curPixX);
							if (edgeImage.at<uchar>(curPixY, curPixX) > 0) {
								r.q = pnew;
								// dot product
								float G_xt = gradientX.at<float>(curPixY, curPixX);
								float G_yt = gradientY.at<float>(curPixY, curPixX);
								mag = sqrt((G_xt * G_xt) + (G_yt * G_yt));
								if (dark_on_light) {
									G_xt = -G_xt / mag;
									G_yt = -G_yt / mag;
								}
								else {
									G_xt = G_xt / mag;
									G_yt = G_yt / mag;

								}

								if (acos(G_x * -G_xt + G_y * -G_yt) < PI / 2.0) {
									float length = sqrt(((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + ((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));

									if ((length > 60) || (length < 2))//笔画宽度大于此值的认为其为不合理的值 20170924
									{
										break;
									}

									for (std::vector<SWTPoint2d>::iterator pit = points.begin(); pit != points.end(); pit++) {
										if (SWTImage.at<float>(pit->y, pit->x) < 0) {
											SWTImage.at<float>(pit->y, pit->x) = length;
										}
										else {
											SWTImage.at<float>(pit->y, pit->x) = std::min(length, SWTImage.at<float>(pit->y, pit->x));
										}
									}
									r.points = points;
									rays.push_back(r);
								}
								break;
							}
						}
					}
				}
				ptr++;
			}
		}

	}



	void histFilter(Mat& img, Mat& labels, Mat& stats, int nLabels)
	{
		//通过连通域的直方图过滤 ST
		Mat histSWT(nLabels, 15, CV_32S);
		//矩阵行 表示连通域，列 该连通域笔画宽度
		histSWT = Mat::zeros(histSWT.size(), CV_32S);
		for (int r = 0; r < img.rows; ++r)
		{
			float* ptr = (float*)img.ptr(r);

			for (int c = 0; c < img.cols; ++c)
			{
				int tmpRow = labels.at<int>(r, c);
				int tmpCol = (int)(*ptr++);
				if (tmpCol != -1)
				{
					//int tmpVal = histSWT.at<int>(tmpRow, tmpCol);
					histSWT.at<int>(tmpRow, tmpCol) = histSWT.at<int>(tmpRow, tmpCol) + 1;
				}

			}
		}

		writeToExcel(img, "imgSWT.xls");
		writeToExcel(histSWT, "histSWT.xls");
		/*
		for (int r = 0; r < histSWT.rows; ++r)
		{
		int validSum = 0;

		for (int c = 0; c < 11; ++c)
		{
		validSum += histSWT.at<int>(r, c);
		}

		}
		*/
		//通过连通域的直方图过滤 END



		//计算每个连通域闭环宽度的均方差 ST
		//利用均方差淘汰连通域的效果不好，原因还需要进一步确认
		std::vector<float> meanSWT(nLabels);

		//计算每个连通域笔画宽度之和

		for (int r = 0; r < img.rows; ++r)
		{
			float* ptr = (float*)img.ptr(r);

			for (int c = 0; c < img.cols; ++c)
			{
				meanSWT[labels.at<int>(r, c)] += (*ptr++);
			}
		}
		//计算每个连通域笔画宽度的均值
		for (int label = 1; label < nLabels; ++label)
		{
			if (stats.at<int>(label, cv::CC_STAT_AREA) > 0)
			{
				meanSWT[label] = meanSWT[label] / stats.at<int>(label, cv::CC_STAT_AREA);
			}
			else
			{
				meanSWT[label] = 10000;
			}
		}
		//均方差
		std::vector<float> meanSTD(nLabels);
		for (int r = 0; r < img.rows; ++r)
		{
			float* ptr = (float*)img.ptr(r);
			for (int c = 0; c < img.cols; ++c)
			{
				int la = labels.at<int>(r, c);
				int mean = meanSWT[la];
				meanSTD[la] += ((*ptr) - mean) * ((*ptr) - mean);

				ptr++;
			}
		}

		std::ofstream fout("meanSTD.xls");

		for (int label = 1; label < nLabels; ++label)
		{
			meanSTD[label] = meanSTD[label] / stats.at<int>(label, cv::CC_STAT_AREA);

			fout.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数  
			fout.precision(2);  // 设置精度 2  

			fout << "label: " << label << "\t" << "meanSTD: " << meanSTD[label] << std::endl;



			/*
			<< "\t" << "AREA: " << stats.at<int>(label, cv::CC_STAT_AREA) << std::endl;

			fout << "CC_STAT_LEFT: " << stats.at<int>(label, cv::CC_STAT_LEFT) << "\t" << "CC_STAT_TOP: " << stats.at<int>(label, cv::CC_STAT_TOP) << std::endl;

			fout << "CC_STAT_WIDTH: " << stats.at<int>(label, cv::CC_STAT_WIDTH) << "\t" << "CC_STAT_HEIGHT: " << stats.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;
			*/
		}
		//计算每个连通域闭环宽度的均方差  END
		//
	}


	//文件名需要添加xls后缀
	void writeToExcel(Mat outputImage, string fileName)
	{

		ofstream Fs(fileName);
		if (!Fs.is_open())
		{
			cout << "error!" << endl;
			return;
		}

		int channels = outputImage.channels();            //获取图像channel  
		int nrows = outputImage.rows;                     //矩阵的行数  
		int ncols = outputImage.cols*channels;             //矩阵的总列数=列数*channel分量数  

		//循环用变量
		int i = 0;
		int j = 0;

		if (outputImage.depth() == CV_8U)//uchar
		{
			for (i = 0; i<nrows; i++)
			{
				for (j = 0; j<ncols; j++)
				{
					int tmpVal = (int)outputImage.ptr<uchar>(i)[j];
					Fs << tmpVal << '\t';
				}
				Fs << endl;
			}
		}
		else if (outputImage.depth() == CV_16S)//short
		{
			for (i = 0; i<nrows; i++)
			{
				for (j = 0; j<ncols; j++)
				{
					Fs << (short)outputImage.ptr<short>(i)[j] << '\t';
				}
				Fs << endl;
			}
		}
		else if (outputImage.depth() == CV_16U)//unsigned short
		{
			for (i = 0; i<nrows; i++)
			{
				for (j = 0; j<ncols; j++)
				{
					Fs << (unsigned short)outputImage.ptr<unsigned short>(i)[j] << '\t';
				}
				Fs << endl;
			}
		}
		else if (outputImage.depth() == CV_32S)//int 
		{
			for (i = 0; i<nrows; i++)
			{
				for (j = 0; j<ncols; j++)
				{
					Fs << (int)outputImage.ptr<int>(i)[j] << '\t';
				}
				Fs << endl;
			}
		}
		else if (outputImage.depth() == CV_32F)//float
		{
			for (i = 0; i<nrows; i++)
			{
				for (j = 0; j<ncols; j++)
				{
					Fs << (float)outputImage.ptr<float>(i)[j] << '\t';
				}
				Fs << endl;
			}
		}
		else//CV_64F double
		{
			for (i = 0; i < nrows; i++)
			{
				for (j = 0; j < ncols; j++)
				{
					Fs << (double)outputImage.ptr<double>(i)[j] << '\t';
				}
				Fs << endl;
			}
		}


		Fs.close();

	}
}