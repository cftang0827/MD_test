#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <stdio.h>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "time.h"

using namespace cv;
using namespace std;
//function declaration
void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color, Point2f& pointBefore , Point2f& pointAfter);
void checkPoint(cv::Mat Image, Point2f middlePoint, vector<Point2f> pointVector , vector<Point2f> &outputVector);
void RGBtoYCbCr(IplImage *);
void SkinColorDetection(IplImage *);
Point drawMidPoint(Mat &inp);
Point2f problemPointBefore = Point2f(380,220);
Point2f problemPointAfter;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    //show original image
    Mat before = imread("C:\\Users\\CF Tang\\github\\MD_test\\build-optical-CFTang-Release\\release\\Ref.jpg");
    Mat after = imread("C:\\Users\\CF Tang\\github\\MD_test\\build-optical-CFTang-Release\\release\\Track.jpg");
    imshow("before",before);
    imshow("after",after);

    //trans image to gray
    Mat before_gray, after_gray;
    cvtColor(before,before_gray,CV_RGB2GRAY);
    cvtColor(after,after_gray,CV_RGB2GRAY);
    imshow("before_gray",before_gray);
    imshow("after_gray",after_gray);

    //blur
    Mat before_gray_blur, after_gray_blur;
    blur(before_gray,before_gray_blur,Size(5, 5), Point(-1, -1));
    blur(after_gray, after_gray_blur,Size(5, 5),Point(-1, -1));
    imshow("before_gray_blur",before_gray_blur);
    imshow("after_gray_blur",after_gray_blur);

    //////////////////////////////try GoodFeatureToTrack//////////////////////////////

    //drawing mask

    Mat masker;
    masker = Mat::zeros(480,640,CV_8UC1);
    Mat beforeTrack;
    Mat afterTrack;

    before_gray.convertTo(beforeTrack,CV_8UC1);
    after_gray.convertTo(afterTrack,CV_8UC1);

    Point point1,point2,point3,point4;
    point1 = cv::Point(250, 180);
    point2 = cv::Point(500, 350);
    point3 = cv::Point(250, 350);
    point4 = cv::Point(500, 180);

    cv::rectangle(masker,point1, point2, cv::Scalar(255, 255, 255), -1);

//    blur(beforeTrack,beforeTrack,Size(21, 21), Point(-1, -1));
//    blur(afterTrack, afterTrack,Size(21, 21),Point(-1, -1));
    cv::GaussianBlur(beforeTrack, beforeTrack, cv::Size(35, 35), 0, 0);
    cv::GaussianBlur(afterTrack, afterTrack, cv::Size(35, 35), 0, 0);


//    imshow("beforeTrack123",beforeTrack);


    //////////////////////////////////////skin detection////////////////////////////////////

        Mat skinBefore;
        Mat skinAfter;
        before.copyTo(skinBefore,masker);
    //    skinBefore = before(row,col);
        imshow("skinBefore",skinBefore);
        after.copyTo(skinAfter,masker);
    //    skinAfter = after(row,col);
        imshow("skinAfter",skinAfter);
        imwrite("skinAfter.jpg",skinAfter);
        imwrite("skinBefore.jpg",skinBefore);

            IplImage* pImgBefore = cvLoadImage("skinBefore.jpg");
            IplImage* pImgAfter = cvLoadImage("skinAfter.jpg");

            IplImage* pImgYCbCr1 = cvCreateImage(cvGetSize(pImgBefore),
                  pImgBefore->depth, pImgBefore->nChannels);

            IplImage* pImgYCbCr2 = cvCreateImage(cvGetSize(pImgAfter),
                  pImgAfter->depth, pImgAfter->nChannels);

            if(pImgBefore)
            {
                cvCopy(pImgBefore, pImgYCbCr1, NULL);
                RGBtoYCbCr(pImgYCbCr1);
                SkinColorDetection(pImgYCbCr1);

                cvShowImage("RGB_Before", pImgBefore);
                cvShowImage("YCbCr_Before", pImgYCbCr1);
            }

            if(pImgAfter)
            {
                cvCopy(pImgAfter, pImgYCbCr2, NULL);
                RGBtoYCbCr(pImgYCbCr2);
                SkinColorDetection(pImgYCbCr2);

                cvShowImage("RGB_After", pImgAfter);
                cvShowImage("YCbCr_After", pImgYCbCr2);

            }

            Mat skinBinaryBefore = Mat(pImgYCbCr1,0);
            Mat skinBinaryAfter = Mat(pImgYCbCr2, 0);

            imshow("skinBinaryBefore",skinBinaryBefore);
            imshow("skinBinaryAfter",skinBinaryAfter);


    //        circle(skinBinaryBefore,drawMidPoint(skinBinaryBefore),4,Scalar(255,9,255),1,8,0);


    //轉成一維mask
            Mat skinBinaryBeforeMask;
            Mat skinBinaryAfterMask;
            skinBinaryBeforeMask = Mat::zeros(480,640,CV_8UC1);
            skinBinaryAfterMask = Mat::zeros(480,640,CV_8UC1);

            cvtColor(skinBinaryBefore,skinBinaryBefore,CV_RGB2GRAY);
            cvtColor(skinBinaryAfter,skinBinaryAfter,CV_RGB2GRAY);

            skinBinaryBefore.convertTo(skinBinaryBeforeMask,CV_8UC1);
            skinBinaryAfter.convertTo(skinBinaryAfterMask,CV_8UC1);





    //goodFeatureToTrack function

    clock_t t;
    t = clock();
    int maxPointSize = 2;
    vector<Point2f> point_before(maxPointSize);
    vector<Point2f> point_after(maxPointSize);

    goodFeaturesToTrack(beforeTrack,point_before,maxPointSize,0.015,10,skinBinaryBeforeMask,3,1,0.04);
    goodFeaturesToTrack(afterTrack,point_after,maxPointSize,0.015,10,skinBinaryAfterMask,3,1,0.04);


    //show image and Feature Point!
    Mat beforeTrackShow, afterTrackShow;
    before.copyTo(beforeTrackShow);
    after.copyTo(afterTrackShow);
    cout<<"beforeTrackShow.at: "<<beforeTrackShow.at<unsigned int>(0,0)<<endl;
    //remove balck point!
    vector<Point2f> pointRemoveBefore;
    vector<Point2f> pointRemoveAfter;

    //feature point
    for(int i = 0;i<point_before.size();i++)
    {
        checkPoint(beforeTrack,point_before[i],point_before,pointRemoveBefore);
    }
    for(int i = 0; i<point_after.size();i++)
    {
        checkPoint(afterTrack,point_after[i],point_after,pointRemoveAfter);
    }

    //draw feature point
    //畫feature point 在before image上
    for(int i=0;i<pointRemoveBefore.size();i++)
    {
        circle(beforeTrackShow,pointRemoveBefore[i],4,Scalar(0,0,0),2,8,0);
    }
    //feature point 在after image
    for(int i=0;i<pointRemoveAfter.size();i++)
    {
        circle(afterTrackShow,pointRemoveAfter[i],4,Scalar(255,255,255),2,8,0);
//        circle(beforeTrackShow,pointRemoveAfter[i],4,Scalar(255,255,255),2,8,0);
    }


    //計算以乳頭為基準的移動向量

    Point2f GFT_Vector = Point2f(problemPointBefore - pointRemoveBefore[0]);
    Point2f problemPointAfterInGoodFeature = pointRemoveAfter[0] + GFT_Vector;
    Point2f Opverall_vector = problemPointAfterInGoodFeature - problemPointBefore;
    cout<<"Opverall_vector = ("<<Opverall_vector.x<<", "<<Opverall_vector.y<<")"<<endl;
    cout<<"problemPointAfterInGoodFeature.x "<<problemPointAfterInGoodFeature.x<<" ,problemPointAfterInGoodFeature.y "<<problemPointAfterInGoodFeature.y<<endl;


    //draw 問題點(已知before問題位置，由上述資料去推論before問題位置會移動多少到after問題位置)

    circle(beforeTrackShow,problemPointBefore,5,Scalar(255,255,0),4,8,0);
    circle(afterTrackShow,problemPointBefore,5,Scalar(255,255,0),4,8,0);
    circle(afterTrackShow,problemPointAfterInGoodFeature,5,Scalar(255,0,255),4,8,0);

    //draw mask on the image showing
    circle(beforeTrackShow,point1,5,Scalar(255,255,255),3,8,0);
    circle(beforeTrackShow,point2,5,Scalar(255,255,255),3,8,0);
    circle(beforeTrackShow,point3,5,Scalar(255,255,255),3,8,0);
    circle(beforeTrackShow,point4,5,Scalar(255,255,255),3,8,0);
    circle(afterTrackShow,point1,5,Scalar(255,255,255),3,8,0);
    circle(afterTrackShow,point2,5,Scalar(255,255,255),3,8,0);
    circle(afterTrackShow,point3,5,Scalar(255,255,255),3,8,0);
    circle(afterTrackShow,point4,5,Scalar(255,255,255),3,8,0);

    t = clock() - t;
    printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

    imshow("beforeTrackShow",beforeTrackShow);
    imshow("afterTrackShow",afterTrackShow);
    //////////////////////////////////////optical flow /////////////////////////////////


    clock_t t1;
    t1 = clock();
    //原始彩色影像
    Mat opticalBeforeRGB,opticalAfterRGB;
    before.copyTo(opticalBeforeRGB);
    after.copyTo(opticalAfterRGB);


//    在opticalBeforeRGB上假設一個問題點 取 Point2f(380,220)

    circle(opticalBeforeRGB,problemPointBefore,5,Scalar(255,0,255),4,8,0);



    //cal optical flow
    Mat flow;

    Mat beforeGrayInMask;
    Mat afterGrayInMask;
    before_gray_blur.copyTo(beforeGrayInMask,skinBinaryBeforeMask);
    after_gray_blur.copyTo(afterGrayInMask,skinBinaryAfterMask);


    calcOpticalFlowFarneback(beforeGrayInMask,afterGrayInMask,flow, 0.5, 3, 10,3, 5, 1.1, 0 );
    Mat cflow;
    Mat beforeGrayInMaskOrigin;
    before_gray.copyTo(beforeGrayInMaskOrigin,skinBinaryBeforeMask);

    cvtColor(beforeGrayInMaskOrigin, cflow, CV_GRAY2BGR);

    drawOptFlowMap(flow, cflow, 5, CV_RGB(0, 0, 255),problemPointBefore, problemPointAfter);
    circle(opticalAfterRGB,problemPointAfter,5,Scalar(255,255,0),4,8,0);
    circle(opticalBeforeRGB,problemPointAfter,5,Scalar(255,255,0),4,8,0);

     imshow("clofw",cflow);
    //problem point test
//    circle(cflow,Point2f(380,220),5,Scalar(255,0,255),4,8,0);


     t = clock() - t;
     printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
    //show image
    imshow("OpticalFlowFarneback", cflow);
    imshow("opticalBeforeRGB",opticalBeforeRGB);
    imshow("opticalAfterRGB",opticalAfterRGB);


    ///////////////////////////////////////matching///////////////////////////////////
    Mat beforeMatching;
    Mat afterMatching;
    Mat beforeInMask;
    Mat afterInMask;

    //beforeMatching.initEmpty();
    //afterMatching.initEmpty();

    //before.convertTo(beforeMatching,CV_RGB2GRAY,1,0);
    //after.convertTo(afterMatching,CV_RGB2GRAY,1,0);

    Range col = Range(250,500);
    Range row = Range(180,350);
    beforeInMask = before_gray(row,col);
    afterInMask = after_gray(row,col);
    imshow("beforeInMask",beforeInMask);
    imwrite("beforeInMask.jpg",beforeInMask);
    imwrite("afterInMask.jpg",afterInMask);
    imshow("afterInMask",afterInMask);

    Mat beforeInMask8UC1;
    Mat afterInMask8UC1;

    beforeInMask.convertTo(beforeInMask8UC1,CV_8UC1);
    afterInMask.convertTo(afterInMask8UC1,CV_8UC1);

    imshow("beforeInMask8UC1",beforeInMask8UC1);
    imshow("afterInMask8UC1",afterInMask8UC1);

//    Mat img_object, img_scene;
//    beforeInMask.convertTo(img_object,CV_8UC1);
//    before_gray.convertTo(img_scene,CV_8UC1);


    /////////////////////////////////Matching///////////////////////////////////////
    Mat img_object = imread("beforeInMask.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_scene = imread( "afterInMask.jpg", CV_LOAD_IMAGE_GRAYSCALE );

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector.detect( img_object, keypoints_object );
    detector.detect( img_scene, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
       { good_matches.push_back( matches[i]); }
    }

    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //test
    cout<<"keypoints_object size "<<keypoints_object.size()<<endl;
    Mat test = Mat(before);

//    for(int i=0;i<keypoints_object.size();i++)
//    {
//        circle(test,Point2f(keypoints_object[i].x,keypoints_object.y),3,Scalar(0,0,0),3,8,0);
//    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, CV_RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );

    waitKey(0);


}



void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color, Point2f& pointBefore, Point2f &pointAfter)
{
    Point2f leftTop;
    Point2f leftDown;
    Point2f rightDown;
    Point2f rightTop;
    if(pointBefore.x-50 > 0 || pointBefore.y-50 > 0 || pointBefore.x+50<cflowmap.cols || pointBefore.y+50<cflowmap.rows)
    {
        circle(cflowmap,pointBefore,5,Scalar(255,0,255),4,8,0);
        leftTop = pointBefore - Point2f(50,50);
        leftDown = Point2f(pointBefore.x-50,pointBefore.y+50);
        rightDown = pointBefore + Point2f(50,50);
        rightTop = Point2f(pointBefore.x+50,pointBefore.y-50);

        rectangle(cflowmap,leftTop,rightDown,Scalar(234,123,111),2,8,0);
    }
    else
    {
        cout<<"error!"<<endl;
    }
    float x_vector = 0;
    float y_vector = 0;
    int pointNum = 0;

    for(int y = 0; y < cflowmap.rows; y += step)
    {
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at< Point2f>(y, x);
            if(fxy.x*fxy.x + fxy.y*fxy.y > 100)
            {
//                cout<<"x "<<x<<", fxy.x "<<fxy.x<<" y "<<y<<", fxy.y "<<fxy.y<<endl;
                line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),color);
                if(x> leftTop.x && y>leftTop.y && x<rightDown.x && y < rightDown.y)
                {
                    pointNum++;
                    x_vector = x_vector + fxy.x;
                    y_vector = y_vector + fxy.y;
                }

            }
        }
    }
    x_vector = x_vector / pointNum;
    y_vector = y_vector / pointNum;
    cout<<"x_vector: "<<x_vector<<endl;
    cout<<"y_vector: "<<y_vector<<endl;

    pointAfter = pointBefore + Point2f(x_vector,y_vector);
    circle(cflowmap,pointAfter,5,Scalar(255,255,0),4,8,0);



}


void checkPoint(cv::Mat Image,cv::Point2f middlePoint, vector<cv::Point2f> pointVector , vector<cv::Point2f>& outputVector)
{
    bool flag;
    for(int a = 0;a<pointVector.size();a++)
    {
        flag = true;
        for(int i = -30 ;i<31;i++)
        {
            for(int j = -30; j<31 ;j++)
            {
                if(Image.at<int>(middlePoint.x+i,middlePoint.y+j) == 0)
                {
                    flag = false;
                    goto endloop;
                    //cout<<"it is false test!!!!"<<endl;
                }
            }

        }
    }

    endloop:
        if(flag)
        {
            //return true;
            outputVector.push_back(middlePoint);

        }else{
            //exit(0);
        }
}



//
int avg_cb = 120;//YCbCr顏色空間膚色cb的平均值
int avg_cr = 155;//YCbCr顏色空間膚色cr的平均值
int skinRange = 22;//YCbCr顏色空間膚色的範圍

void RGBtoYCbCr(IplImage *image)
{
    CvScalar scalarImg;//原始RGB影像的3 channel
    double cb, cr, y;
    for( int i = 0; i < image->height; i++ )
    for( int j = 0; j < image->width; j++ )
    {
        scalarImg = cvGet2D(image, i, j);//從影像中取RGB值
        //把RGB轉換成Y Cb Cr依照不同係數
        y =  (16 + scalarImg.val[2]*0.257 + scalarImg.val[1]*0.504
            + scalarImg.val[0]*0.098);
        cb = (128 - scalarImg.val[2]*0.148 - scalarImg.val[1]*0.291
            + scalarImg.val[0]*0.439);
        cr = (128 + scalarImg.val[2]*0.439 - scalarImg.val[1]*0.368
            - scalarImg.val[0]*0.071);
        cvSet2D(image, i, j, cvScalar( y, cr, cb));//最後再把YCbCr color space mapping to image
    }
}

void SkinColorDetection(IplImage *image)
{
    CvScalar scalarImg;
    double cb, cr;
    for( int i = 0; i < image->height; i++ )
    for( int j = 0; j < image->width; j++ )
    {
        scalarImg = cvGet2D(image, i, j);
        cr = scalarImg.val[1];
        cb = scalarImg.val[2];
        if((cb > avg_cb-skinRange && cb < avg_cb+skinRange) &&
                  (cr > avg_cr-skinRange && cr < avg_cr+skinRange))
        cvSet2D(image, i, j, cvScalar( 255, 255, 255));
        else
            cvSet2D(image, i, j, cvScalar( 0, 0, 0));
    }
}

Point drawMidPoint(Mat& inp)
{
    Vec3b color;
    int x = 0;
    int y = 0;
    int pointNum = 0;
    for(int i = 0;i<inp.cols ; i++)
    {
        for(int j = 0; j<inp.rows ; j++)
        {
            color = inp.at<Vec3b>(Point(i,j));
            if(color[0] == 1&& color[1] == 1&& color[2] == 1)
            {
                x = x+i;
                y = y+i;
                pointNum++;
            }
        }
    }

    int xAv = x/pointNum;
    int yAv = y/pointNum;
//    cout<<"x = "<<xAv<<", "<<"y = "<<yAv<<endl;
    Point draw = Point(xAv,yAv);

    return draw;
//    circle(inp,draw,5,Scalar(255,0,255),3,8,0);
}






