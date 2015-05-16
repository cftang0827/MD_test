#-------------------------------------------------
#
# Project created by QtCreator 2014-08-29T13:55:15
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = relocation
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    D:/opencv/build/include/opencv/cv.h

FORMS    += mainwindow.ui

##-------------------------------------------------
##
## OpenCV lib include
##
##-------------------------------------------------



#INCLUDEPATH += D:\opencv\build\include \
#LIBS += -LD:\opencv\build\x86\vc11\lib\
#    -lopencv_core246.lib \
#    -lopencv_highgui246.lib \
#    -lopencv_imgproc246.lib \
#    -lopencv_features2d246.lib \
#    -lopencv_calib3d246.lib \

INCLUDEPATH += D:\opencv\build\include\

CONFIG(release,debug|release){
    LIBS += D:\opencv\build\x86\vc11\lib\opencv_calib3d246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_contrib246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_core246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_features2d246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_flann246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_gpu246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_highgui246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_imgproc246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_legacy246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_ml246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_objdetect246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_ts246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_video246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_nonfree246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_flann246.lib
}

CONFIG(debug,debug|release){
    LIBS += D:\opencv\build\x86\vc11\lib\opencv_calib3d246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_contrib246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_core246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_features2d246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_flann246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_gpu246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_highgui246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_imgproc246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_legacy246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_ml246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_objdetect246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_ts246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_video246d.lib \
            D:\opencv\build\x86\vc11\lib\opencv_nonfree246.lib \
            D:\opencv\build\x86\vc11\lib\opencv_flann246.lib

}
