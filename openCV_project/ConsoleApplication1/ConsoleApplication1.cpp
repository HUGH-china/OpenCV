#include <iostream>
#include<cmath>
#include<algorithm>
#include <opencv2\core\core.hpp> 
#include <opencv2\imgproc\imgproc.hpp> 
#include <opencv2\highgui\highgui.hpp> 
using namespace std;
using namespace cv;
int pc=0;
Point2f point[10];//用来存放畸变图像的选择约束点
//平移函数 以图像的左上角为原点 
Mat Move(int y, int x, Mat image) //行,列,图像
{
	
	//存放平移后的结果 左正右负
	Mat result= Mat::zeros(image.size(),image.type());
	int row, col;
	row = image.rows;
	col = image.cols;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			int temp_y = i + y;
			int temp_x = j + x;
			if (temp_y >= 0 && temp_x >= 0 && temp_x < col && temp_y < row)
			{
				//此时认为result图像在该点有像素信息映射
				result.at<Vec3b>(i, j) = image.at<Vec3b>(temp_y,temp_x);
			}
			else
			{
				result.at<Vec3b>(i, j) = Vec3b(0,0,0);
			}
		}
	}
	return result;
}
Mat resize(Mat image, float ky,float kx)//ky,kx浮点类型记录缩放倍数
{
	int size_y = image.rows * ky;
	int size_x = image.cols * kx;
	Mat result = Mat::zeros(size_y, size_x, image.type());
		for (int i = 0; i < size_y; i++)
		{
			for (int j = 0; j < size_x; j++)
			{
				float fy = i / ky;
				float fx = j / kx;
				int y0 = int(fy);
				int x0 = int(fx);
				float q = fy - y0;
				float p = fx - x0;
				int y1 = min(y0 + 1, image.rows - 1);
				int x1 = min(x0 + 1, image.cols - 1);
				Vec3b temp_1, temp_2, temp_3, temp_4;
				temp_1 = image.at<Vec3b>(y0, x0);
				temp_2 = image.at<Vec3b>(y0, x1);
				temp_3 = image.at<Vec3b>(y1, x0);
				temp_4 = image.at<Vec3b>(y1, x1);
				Vec3b temp_reuslt;
					for (int k = 0; k < 3; k++)
					{
						float val = (1 - q) * (1 - p) * temp_1[k] + (1 - q) * p * temp_2[k]
							+ q * (1 - p) * temp_3[k] + q * p * temp_4[k];
						temp_reuslt[k] = saturate_cast<uchar>(val);
					}
				result.at<Vec3b>(i, j) = temp_reuslt;
			}
		}
		return result;
}
Mat rotateImage(Mat image ,int n)// n为旋转度数
{
	int nrow = image.rows;
	int ncol = image.cols;
	Point2f center(ncol / 2, nrow / 2);
	Mat rotmat = getRotationMatrix2D(center, n, 1);
	double cosVal = abs(rotmat.at<double>(0, 0));
	double sinVal = abs(rotmat.at<double>(0, 1));
	int new_col = int(cosVal * ncol + sinVal * nrow);
	int new_row = int(sinVal * ncol + cosVal * nrow);
	rotmat.at<double>(0, 2) += (new_col / 2 - center.x);
	rotmat.at<double>(1, 2) += (new_row / 2 - center.y);
	Mat dst(new_row, new_col, image.type());
	warpAffine(image, dst, rotmat, dst.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
	return dst;
}
// 鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* userdata)
{

	if (event == EVENT_LBUTTONDOWN) // 左键点击
	{
		Mat* img = reinterpret_cast<Mat*>(userdata);
		cout << x <<" " << y;
		cout << endl;	
		point[pc] = Point2f(x, y);
		pc++;
	}
	
}
Mat rebuild(Mat image,int new_row,int new_col)
{
	//创建目标图片的4个约束点
	Point2f dstPoint[10];
	dstPoint[0] = Point2f(0, 0);
	dstPoint[1] = Point2f(new_col-1, 0);
	dstPoint[2] = Point2f(new_col - 1, new_row - 1);
	dstPoint[3] = Point2f(0, new_row - 1);
	//建立几何模型
	Mat H = getPerspectiveTransform(point, dstPoint);
	Mat dst;
	dst = Mat::zeros(new_row, new_col, image.type());
	warpPerspective(image, dst, H, dst.size());
	return dst;
}
int main()
{
	Mat srcImg;
	srcImg = imread("..\\test3.jpg");  // 读入原始图像 
	if (srcImg.empty()) {
		cerr << "Failed to load image file" << endl;
		return -1;
	}
	//Mat result=Move(50, 100, srcImg);//左正右负
	//imshow("移动后的图像", result);
	//imshow("原图", srcImg);
	//Mat result = resize(srcImg, 0.5, 0.5);
	//imshow("缩放的图像", result);
	//imshow("原图", srcImg);
	//Mat result= rotateImage(srcImg,45);
	//imshow("旋转后的图像", result);
	//imshow("原图", srcImg);
	namedWindow("Image", WINDOW_AUTOSIZE);
	setMouseCallback("Image", onMouse, &srcImg);
	imshow("Image", srcImg);
	waitKey();
	for (int i = 0; i < pc; i++)
		cout << point[i];
	Mat result = rebuild(srcImg, 512, 512);
	//###vital!!选择图像的边界点从(0,0)点顺时针选择!!!!#######
	imshow("几何畸变校正后的图像", result);
	imshow("原图", srcImg);
	waitKey();
	return 0;
}
