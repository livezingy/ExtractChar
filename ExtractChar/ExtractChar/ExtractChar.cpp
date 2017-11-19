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

#include "stdafx.h"
#include <cassert>
#include <fstream>
#include <exception>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "extraction.h"

using namespace std;
using namespace cv;
using namespace extraction;

Mat loadByteImage(const char * name) 
{
	Mat image = imread(name);

	if (image.empty()) 
	{
		return Mat();
	}
	cvtColor(image, image, CV_BGR2RGB);
	return image;
}

int mainTextDetection(int argc, char* imgPath)
{//char** argv) 
	
	Mat byteQueryImage = loadByteImage(imgPath);
	if (byteQueryImage.empty()) 
	{
		cerr << "couldn't load query image" << endl;
		return -1;
	}

	// Detect text in the image
	textDetection(byteQueryImage, atoi("0"));
	

	return 0;
}



int _tmain(int argc, _TCHAR* argv[])
{
	mainTextDetection(1, "Test/9.png");
	return 0;
}

