#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdlib.h>

#define AVX2_ENABLED

int optimizedBlockMatchScalar(cv::Mat& leftMat, cv::Mat& rightMat, cv::Mat& outputMat, int blockSize, int numDisparities);
#ifdef AVX2_ENABLED
int optimizedBlockMatchAVX2(cv::Mat& leftMat, cv::Mat& rightMat, cv::Mat& outputMat, int blockSize, int numDisparities);
#endif

int main();