// StereoOptimizedSmartMemory.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdlib.h>

int optimizedBlockMatchScalar(cv::Mat &leftMat, cv::Mat &rightMat, cv::Mat &outputMat, int blockSize, int numDisparities) {
    /*
    * Stereo block matching algorithm, laid out for vector operations but no SIMD done (in this implementation)
    */
    if (leftMat.rows != rightMat.rows || leftMat.cols != rightMat.cols) {
        // this algorithm assumes both images are of the same size
        return -1;
    }
    uchar* left = leftMat.isContinuous() ? leftMat.data : leftMat.clone().data;
    uchar* right = leftMat.isContinuous() ? rightMat.data : rightMat.clone().data;
    int blockRad = (blockSize - 1) / 2;
    int numNonzeroDisparities = numDisparities - 1;
    int inWidth = leftMat.cols;
    int inHeight = leftMat.rows;
    int outWidth = inWidth - 2 * blockRad - numNonzeroDisparities;
    int outHeight = inHeight - 2 * blockRad;
    int diffHeight = inWidth - numNonzeroDisparities;
    int xnHeight = diffHeight - 2 * blockRad; // height of x1 and x2, also equals outWidth
    int x1SubLength = xnHeight * numDisparities;
    outputMat = cv::Mat(outHeight, outWidth, CV_16UC1);
    uint16_t* output = (uint16_t*)outputMat.data;

    // I'm using calloc teehee (hope I don't memleak)
    // diff stores the absolute differences between compared pixels
    // to start, diff[i][j] = abs((int)left[0][i] - (int)right[0][i+j])
    // width: numDisparities. height: diffHeight
    uint8_t* diff = (uint8_t*)calloc(numDisparities * diffHeight, sizeof(uint8_t));
    memset(diff, 0, sizeof(diff));
    // x1 stores a vertical rolling sum of diff, which creates the horizontal part of the block
    // x1 also is an array of arrays to manage the rolling sum nature of the vertical part of the block
    // x1[l][i][j] = sum(diff[k][j] for k in range(i:i+blockSize))
    uint16_t* x1 = (uint16_t*)calloc(numDisparities * xnHeight * blockSize, sizeof(uint16_t));
    memset(x1, 0, sizeof(x1));
    // x2 stores the vertical rolling sum of x1, so it is a single array instead of an AoA like x1 is
    // otherwise, it has the same dimensions since it's a straight sum, but it's in a bigger number format
    uint32_t* x2 = (uint32_t*)calloc(numDisparities * xnHeight, sizeof(uint32_t));
    memset(x2, 0, sizeof(x2));

    for (int y = 0; y < outHeight; y++) {
        if (y == 0) {
            // start by filling the buffers, then we can go efficiently after that
            // loop through the vertical block, creating a good x1 and x2
            for (int i = 0; i < blockSize; i++) {
                // populate each row of diff
                for (int l = 0; l < diffHeight; l++) {
                    // populate each column of diff
                    for (int j = 0; j < numDisparities; j++) {
                        diff[l * numDisparities + j] = (uint8_t)abs((int)right[i * inWidth + l] - (int)left[i * inWidth + l + j]);
                    }
                }
                // calculate first line of ith subarray of x1 (get the rolling sum rolling so to speak)
                for (int j = 0; j < blockSize; j++) {
                    // add each of the first blockSize lines of diff to the first line of x1
                    for (int k = 0; k < numDisparities; k++) {
                        x1[i * x1SubLength + k] += diff[j * numDisparities + k];
                    }
                }
                // now calculate the efficient rolling sum of the x1 subarray
                for (int j = 0; j < xnHeight - 1; j++) {
                    // loop along columns of x1
                    for (int k = 0; k < numDisparities; k++) {
                        // start with previous x1, subtract old diff, add new diff
                        x1[i * x1SubLength + (j + 1) * numDisparities + k] = x1[i * x1SubLength + j * numDisparities + k] - diff[j * numDisparities + k] + diff[(j + blockSize) * numDisparities + k];
                    }
                }
                // now accumulate x1 subarray into x2
                for (int j = 0; j < x1SubLength; j++) {
                    x2[j] += x1[i * x1SubLength + j];
                }
            }
        }
        // main loop
        // first, create outputs
        // run argmin across rows of x2
        uint32_t minval;
        uint16_t minind;
        for (int i = 0; i < outWidth; i++) {
            minval = x2[i * numDisparities];
            minind = 0;
            for (int j = 1; j < numDisparities; j++) {
                if (x2[i * numDisparities + j] < minval) {
                    minval = x2[i * numDisparities + j];
                    minind = j;
                }
            }
            output[y * outWidth + i] = minind;
        }
        // now, exit loop if we are done
        if (y == outHeight - 1) {
            break;
        }
        // next, update x2 (roll the rolling sum)
        // first, subtract the old x1 subarray
        for (int i = 0; i < xnHeight; i++) {
            for (int j = 0; j < numDisparities; j++) {
                x2[i * numDisparities + j] -= x1[(y % blockSize) * x1SubLength + i * numDisparities + j];
            }
        }
        // now, create the new x1
        // start by zeroing the new x1 segment
        memset(x1 + (y % blockSize) * x1SubLength, 0, x1SubLength * sizeof(uint16_t));
        // populate each column of diff
        for (int l = 0; l < diffHeight; l++) {
            // populate each row of diff
            for (int j = 0; j < numDisparities; j++) {
                diff[l * numDisparities + j] = (uint8_t)abs((int)right[(y + blockSize) * inWidth + l] - (int)left[(y + blockSize) * inWidth + l + j]);
            }
        }
        // calculate first line of (y % blockSize)th subarray of x1 (get the rolling sum rolling so to speak)
        for (int j = 0; j < blockSize; j++) {
            // add each of the first blockSize lines of diff to the first line of x1
            for (int k = 0; k < numDisparities; k++) {
                x1[(y % blockSize) * x1SubLength + k] += diff[j * numDisparities + k];
            }
        }
        // now calculate the efficient rolling sum of the x1 subarray
        for (int j = 0; j < xnHeight - 1; j++) {
            // loop along rows of x1
            for (int k = 0; k < numDisparities; k++) {
                // start with previous x1, subtract old diff, add new diff
                x1[(y % blockSize) * x1SubLength + (j + 1) * numDisparities + k] = x1[(y % blockSize) * x1SubLength + j * numDisparities + k] - diff[j * numDisparities + k] + diff[(j + blockSize) * numDisparities + k];
            }
        }
        // now accumulate x1 subarray into x2
        for (int j = 0; j < x1SubLength; j++) {
            x2[j] += x1[(y % blockSize) * x1SubLength + j];
        }
        
    }
    free(diff);
    free(x1);
    free(x2);
    return 0;
}

int main()
{
    cv::Mat left = cv::imread("C:/Users/ahpin/Documents/opencv_utilities/left.jpg",cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread("C:/Users/ahpin/Documents/opencv_utilities/right.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat disparity, disparity_uint8;
    std::chrono::system_clock::time_point before = std::chrono::system_clock::now();
    optimizedBlockMatchScalar(left, right, disparity, 15, 96);
    std::chrono::system_clock::time_point after = std::chrono::system_clock::now();
    std::chrono::microseconds elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
    std::cout << "stereo took " << elapsed_us.count() << " microseconds.\n";
    disparity.convertTo(disparity_uint8, CV_8U);
    cv::imshow("left", left);
    cv::imshow("right", right);
    cv::imshow("disparity", disparity_uint8);
    cv::waitKey(0);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
