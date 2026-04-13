#include <opencv2\core\core.hpp> 
#include <opencv2\imgproc\imgproc.hpp> 
#include <opencv2\highgui\highgui.hpp> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include<cmath>
#include<algorithm>
using namespace std;
using namespace cv;
Mat rise_light_line(Mat image,Point2f a,Point2f b)
{
    
    Mat result = Mat::zeros(image.size(), image.type());
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            float value = (uchar)image.at<uchar>(i, j);
            if (value <= a.x&&value>=0)
            {
                value = value;
            }
            if (value <= b.x && value >= a.x)
            {
                value = (b.y - a.y) / (b.x - a.x) * (value - a.x) + a.y;
            }
            if (value >= b.x && value <= 255)
            {
                value = (255 - b.y) / (255 - b.x) * (value - b.x) + b.y;
            }
            result.at<uchar>(i, j) = int(value);
        }
    }
    return result;
}
Mat rise_light_log(Mat image)
{
    
    Mat result = Mat::zeros(image.size(), image.type());
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            float value = image.at<uchar>(i, j);
            value = 255 / log10(256) * log(value + 1);//曝光严重
            result.at<uchar>(i, j) = int(value);
        }
    }
    return result;
}
Mat HE(Mat image)//直方图均衡化
{
    Mat dst = Mat::zeros(image.size(), image.type());
    float F[256] = {0};//记录概率分布函数
    struct Node
    {
        int f=0;//用于记录原图每一个色度值的概率函数
        int changed_color=NULL;
    };
    Node color[256] = {};
    int N = image.rows * image.cols;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            uchar value = image.at<uchar>(i, j);
            color[value].f++;
        }
    }
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            F[i] += color[j].f;
        }
        F[i] = F[i] / N;
        //重新分配颜色分布
        int a;
        a = F[i] * 255 + 0.5;
        color[i].changed_color = a;
    }
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            uchar value = image.at<uchar>(i, j);
            value = color[value].changed_color;
            dst.at<uchar>(i, j) = value;
        }
    }
    return dst;
}
Mat function_one(Mat image,int type)//平滑处理
{
    Mat dst = Mat::zeros(image.size(), image.type());
    //邻域平滑算子
    int a[3][3] = { {1,1,1},
                    {1,1,1},
                    {1,1,1} };
    //高斯平滑算子
    int b[3][3] = { {1,2,1},
                    {2,4,2},
                    {1,2,1} };
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            uchar value;
            int result = 0;
            value = image.at<uchar>(i, j);
            int sum = 0;
            if (type == 0)//邻域平滑处理
            {
                for (int ni = 0; ni < 3; ni++)
                {
                    for (int nj = 0; nj < 3; nj++)
                    {
                        int r = i + ni - 1;
                        int c = j + nj - 1;
                        if (r >= 0 && r < image.rows && c >= 0 && c < image.cols)
                        {
                            sum += a[ni][nj];
                            result += a[ni][nj] * image.at<uchar>(r, c);
                        }
                    }
                }
                result = result / sum;
            }
            if (type == 1)//高斯平滑定理
            {
                result = 0;
                int sum = 0;
                for (int ni = 0; ni < 3; ni++)
                {
                    for (int nj = 0; nj < 3; nj++)
                    {
                        int r = i + ni - 1;
                        int c = j + nj - 1;
                        if (r >= 0 && r < image.rows && c >= 0 && c < image.cols)
                        {
                            sum += b[ni][nj];
                            result += b[ni][nj] * image.at<uchar>(r, c);
                        }
                    }
                }
                result = result / sum;
            }
            dst.at<uchar>(i, j) = result;
        }
    }
    return dst; 
}
Mat function_MedianFilter(Mat image,int n)//中值滤波平滑,n表示窗口的大小n<10
{
    Mat dst = Mat::zeros(image.size(), image.type());
    vector<int> f;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int r = i - (n - 1) / 2;
            int c = j - (n - 1) / 2;
            for (int ni = 0; ni < n; ni++)
            {
                for (int nj = 0; nj < n; nj++)
                {
                    int nr=r+ni, nc=c+nj;
                    nr = max(0, min(image.rows - 1, nr));
                    nc = max(0, min(image.cols - 1, nc));
                    f.push_back(image.at<uchar>(nr, nc));
                }
            }
            uchar value = 0;
            sort(f.begin(), f.end());
            if ((n * n) % 2 == 0)
            {
                int ans;
                ans = n * n / 2;
                value = (f[ans-1] + f[ans]) / 2;
            }
            if ((n * n % 2 == 1))
            {
                int ans;
                ans = ((n * n) + 1) / 2;
                value = f[ans - 1];
            }
            dst.at<uchar>(i, j) = value;
            f.clear();
        }
    }
    return dst;
}
int main()
{
    Mat srcImg;
    srcImg = imread("..\\LenaSaltNoise.png", IMREAD_GRAYSCALE);
    if (srcImg.empty())
    {
        cout << "无法打开图像" << endl;
        return -1;
    }
    imshow("原图", srcImg);
    Mat result;
    //接口测试
    //Point2f a(30, 30);
    //Point2f b(60, 155);
    //Mat result = rise_light_line(srcImg, a, b);
    //imshow("增亮后的图像", result);
    //Mat result = rise_light_log(srcImg);
    //imshow("增强后的图像", result);
    //Mat result = HE(srcImg);
    //imshow("直方图均衡化的结果", result);

    //for (int i = 0; i < 10; i++)
    //{
    //    //进行多次滤波但是效果不是特别理想
    //   result = function_one(srcImg, 0);
    //}
    //imshow("平滑处理后的图像", result);
    result = function_MedianFilter(srcImg, 3);
    imshow("中值滤波处理后的noise图像", result);
    waitKey();
    return 0;
}