#include <stdio.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cv.h>	
#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv.hpp>
#include <io.h>
#include <fstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

int _PriorSize = 15;		//窗口大小 15
double _topbright = 0.001;//亮度最高的像素比例
double _w = 0.95;			//w0.95
float t0 = 0.1;			//T(x)的最小值   因为不能让tx小于0 等于0 效果不好   0.1
int SizeH = 0;			//图片高度
int SizeW = 0;			//图片宽度
int SizeH_W = 0;			//图片中的像素总数 H*W
Vec<float, 3> a;//全球大气的光照值
Mat trans_refine;
Mat dark_out1;

int rows, cols;
const double w = 0.95;
const int r = 7;

vector<string> files;//文件夹下各文件名

char img_name[100] = "1.png";

int inverse(Mat& src) {
	Mat gray_src;
	//	imshow("原图", src);
	Mat dst;
	dst.create(src.size(), src.type());
	int height = src.rows;
	int width = src.cols;
	int nc = src.channels();
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//判断如果通道为1则直接取反色
			if (nc == 1) {
				int gray = src.at<uchar>(row, col);
				dst.at<uchar>(row, col) = 255 - gray;
			}
			//判断如果通道为3则读取各个颜色通道的像素值
			//Vec3b表示3通道uchar类型的数据
			else if (nc == 3) {
				int b = src.at<Vec3b>(row, col)[0];
				int g = src.at<Vec3b>(row, col)[1];
				int r = src.at<Vec3b>(row, col)[2];
				dst.at<Vec3b>(row, col)[0] = 255 - b;
				dst.at<Vec3b>(row, col)[1] = 255 - g;
				dst.at<Vec3b>(row, col)[2] = 255 - r;
			}
		}
	}
	//	imshow("反色之后", dst);
	src = dst;
	waitKey(0);
	return 0;
}

//计算暗通道
//J^{dark}(x)=min( min( J^c(y) ) )

Mat DarkChannelPrior(Mat img)
{
	Mat dark = Mat::zeros(img.rows, img.cols, CV_32FC1);
	//新建一个所有元素为0的单通道的矩阵

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			dark.at<float>(i, j) = min(
				min(img.at<Vec<float, 3>>(i, j)[0], img.at<Vec<float, 3>>(i, j)[1]),
				min(img.at<Vec<float, 3>>(i, j)[0], img.at<Vec<float, 3>>(i, j)[2])
			);//就是两个最小值的过程
		}
	}
	erode(dark, dark_out1, Mat::ones(_PriorSize, _PriorSize, CV_32FC1));
	//这个函数叫腐蚀 做的是窗口大小的模板运算 ,对应的是最小值滤波,即 黑色图像中的一块块的东西

	return dark_out1;//这里dark_out1用的是全局变量
}

void printMatInfo(char* name, Mat m)
{
	cout << name << ":" << endl;
	cout << "\t" << "cols=" << m.cols << endl;
	cout << "\t" << "rows=" << m.rows << endl;
	cout << "\t" << "channels=" << m.channels() << endl;
}

//计算全球大气光强A
//src为输入的带雾图像，darkChannelImg为暗通道图像，r为最小值滤波的窗口半径

Mat getDarkChannelImg(const Mat src, const int r) {
	int height = src.rows;
	int width = src.cols;
	Mat darkChannelImg(src.size(), CV_8UC1);
	Mat darkTemp(darkChannelImg.size(), darkChannelImg.type());

	//求取src中每个像素点三个通道中的最小值，将其赋值给暗通道图像中对应的像素点
	for (int i = 0; i < height; i++) {
		const uchar* srcPtr = src.ptr<uchar>(i);
		uchar* dstPtr = darkTemp.ptr<uchar>(i);

		for (int j = 0; j < width; j++) {
			int b = srcPtr[3 * j];
			int g = srcPtr[3 * j + 1];
			int r = srcPtr[3 * j + 2];
			dstPtr[j] = min(min(b, g), r);
		}
	}
	//把图像分成patch,求patch框内的最小值,得到dark_channel image
	//r is the patch radius, patchSize=2*r+1 
	//这一步实际上是最小值滤波的过程
	cv::Mat rectImg;
	int patchSize = 2 * r + 1;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			cv::getRectSubPix(darkTemp, cv::Size(patchSize, patchSize), cv::Point(i, j), rectImg);
			double minValue = 0;
			cv::minMaxLoc(rectImg, &minValue, 0, 0, 0); //get min pix value
			darkChannelImg.at<uchar>(j, i) = cv::saturate_cast<uchar>(minValue);//using saturate_cast to set pixel value to [0,255]  
		}
	}

	return darkChannelImg;
}

struct node
{
	int x, y, val;
	node() {}
	node(int _x, int _y, int _val) :x(_x), y(_y), val(_val) {}

	bool operator<(const node& rhs)
	{
		return val > rhs.val;
	}
};

//估算全局大气光值
int getGlobalAtmosphericLightValue(Mat darkChannel, cv::Mat img, bool meanMode = false, float percent = 0.001) {
	int size = rows * cols;
	std::vector <node> nodes;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			node tmp;
			tmp.x = i, tmp.y = j, tmp.val = darkChannel.at<uchar>(i, j);
			nodes.push_back(tmp);
		}
	}

	sort(nodes.begin(), nodes.end());
	int atmosphericLight = 0;

	if (int(percent * size) == 0)
	{
		for (int i = 0; i < 3; i++)
		{
			if (img.at<Vec3b>(nodes[0].x, nodes[0].y)[i] > atmosphericLight)
			{
				atmosphericLight = img.at<Vec3b>(nodes[0].x, nodes[0].y)[i];
			}
		}
	}

	//开启均值模式
	if (meanMode == true)
	{
		int sum = 0;
		for (int i = 0; i < int(percent * size); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				sum = sum + img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
			}
		}
	}

	//获取暗通道在前0.1%的位置的像素点在原图像中的最高亮度值
	for (int i = 0; i < int(percent * size); i++) {
		for (int j = 0; j < 3; j++) {
			if (img.at<Vec3b>(nodes[i].x, nodes[i].y)[j] > atmosphericLight) {
				atmosphericLight = img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
			}
		}
	}

	return atmosphericLight;
}
/*
double getGlobalAtmosphericLightValue(Mat darkChannel, cv::Mat img, bool meanMode = false, float percent = 0.001) {
	double minAtomsLight = 249;//经验值
	double maxValue = 0;
	cv::Point maxLoc;
	minMaxLoc(darkChannel, NULL, &maxValue, NULL, &maxLoc);
	double A = min(minAtomsLight, maxValue);
	return A;
}
*/
Mat getTransimissionImg(const Mat darkChannelImg, const double A)
{
	cv::Mat transmissionImg(darkChannelImg.size(), CV_8UC1);
	cv::Mat look_up(1, 256, CV_8UC1);

	uchar* look_up_ptr = look_up.data;
	for (int k = 0; k < 256; k++)
	{
		look_up_ptr[k] = cv::saturate_cast<uchar>(255 * (1 - w * k / A));
	}

	cv::LUT(darkChannelImg, look_up, transmissionImg);

	return transmissionImg;
}

Mat getDehazedImg(const Mat src, const Mat transmissionImage, const int A)
{
	double tmin = 0.1;
	double tmax = 0;

	Vec3b srcData;
	Mat dehazedImg = Mat::zeros(src.size(), CV_8UC3);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			double transmission = transmissionImage.at<uchar>(i, j);
			srcData = src.at<Vec3b>(i, j);

			tmax = max(transmission / 255, tmin);
			//(I-A)/t +A  
			for (int c = 0; c < 3; c++)
			{
				dehazedImg.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(abs((1.0 * srcData.val[c] - A) / tmax + A));
			}
		}
	}
	return dehazedImg;
}

void get_files() {
	long long hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	hFile = _findfirst("jujing", &fileinfo);
	while (_findnext(hFile, &fileinfo) == 0) {
		files.push_back(fileinfo.name);
	}
	_findclose(hFile);
}

cv::Mat GuidedFilter(cv::Mat& I, cv::Mat& p, int r, double eps) {
	int wsize = 2 * r + 1;
	//数据类型转换
	I.convertTo(I, CV_64F, 1.0 / 255.0);
	p.convertTo(p, CV_64F, 1.0 / 255.0);

	//meanI=fmean(I)
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//meanP=fmean(P)
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//corrI=fmean(I.*I)
	cv::Mat mean_II;
	mean_II = I.mul(I);
	cv::boxFilter(mean_II, mean_II, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//corrIp=fmean(I.*p)
	cv::Mat mean_Ip;
	mean_Ip = I.mul(p);
	cv::boxFilter(mean_Ip, mean_Ip, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//varI=corrI-meanI.*meanI
	cv::Mat var_I, mean_mul_I;
	mean_mul_I = mean_I.mul(mean_I);
	cv::subtract(mean_II, mean_mul_I, var_I);

	//covIp=corrIp-meanI.*meanp
	cv::Mat cov_Ip;
	cv::subtract(mean_Ip, mean_I.mul(mean_p), cov_Ip);

	//a=conIp./(varI+eps)
	//b=meanp-a.*meanI
	cv::Mat a, b;
	cv::divide(cov_Ip, (var_I + eps), a);
	cv::subtract(mean_p, a.mul(mean_I), b);

	//meana=fmean(a)
	//meanb=fmean(b)
	cv::Mat mean_a, mean_b;
	cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//q=meana.*I+meanb
	cv::Mat q;
	q = mean_a.mul(I) + mean_b;

	//数据类型转换
	I.convertTo(I, CV_8U, 255);
	p.convertTo(p, CV_8U, 255);
	q.convertTo(q, CV_8U, 255);

	return q;

}


signed main() {
//	get_files();

	//for (int i = 0; i < files.size(); ++i) {
		cv::Mat src, darkChanelImg;
		src = cv::imread("lujing", 1);
		rows = src.rows;
		cols = src.cols;
		if (!src.data) std::cout << "Load Image Error!";

		inverse(src);
		src.convertTo(src, CV_8U, 1, 0);

		darkChanelImg = getDarkChannelImg(src, r);
		//imshow("darkChanelImg", darkChanelImg);

		double A = getGlobalAtmosphericLightValue(darkChanelImg, src);

		Mat transmissionImage(darkChanelImg.size(), darkChanelImg.type());
		transmissionImage = getTransimissionImg(darkChanelImg, A);
		//imshow("transmissionImage", transmissionImage);

		Mat dehazedImg = Mat::zeros(src.rows, src.cols, CV_8UC3);

		double t = (double)getTickCount();
		dehazedImg = getDehazedImg(src, transmissionImage, A);
		t = (double)getTickCount() - t;
		//std::cout << 1000 * t / (getTickFrequency()) << "ms" << std::endl;

		inverse(src);

		inverse(dehazedImg);

		cv::Mat dst1, dehazedImg_input, I;
		dehazedImg.copyTo(dehazedImg_input);
		if (dehazedImg.channels() > 1)
			cv::cvtColor(dehazedImg, I, CV_RGB2GRAY); //若引导图为彩色图，则转为灰度图
		std::vector<cv::Mat> p, q;
		if (dehazedImg.channels() > 1) {             //输入为彩色图
			cv::split(dehazedImg_input, p);
			for (int i = 0; i < dehazedImg.channels(); ++i) {
				dst1 = GuidedFilter(I, p[i], 4, 0.1 * 0.1);
				q.push_back(dst1);
			}
			cv::merge(q, dst1);
		}
		else {                               //输入为灰度图
			dehazedImg.copyTo(I);
			dst1 = GuidedFilter(I, dehazedImg_input, 4, 0.1 * 0.1);
		}
		//imshow("src", src);
		imshow("src", src);
		imshow("dehazedImg", dehazedImg);

		cvWaitKey(0);
	//}
	return 0;
}