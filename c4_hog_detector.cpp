#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
 
using namespace cv;
using namespace std;
 
//円周率
const double PI = 3.141592;
// 16px * 16px
// const vector<int> CELL_SIZE = { 16, 16 };
const int CELL_SIZE[2] = { 16, 16 };
// 2cell * 2cell
// const vector<int> BLOCK_SIZE = { 2, 2 };
const int BLOCK_SIZE[2] = { 2, 2 };
//0度 - 160度
const int GRADIENT_SIZE = 9;
 
int main(){
    //画像
    Mat img = imread( "google.jpg", CV_LOAD_IMAGE_GRAYSCALE );
    Mat numImg = Mat::zeros( img.rows, img.cols, CV_64F );
 
    //power law (gamma) equalization
    int max = 0;
    int min = 255;
    for ( int y = 0; y < img.rows; y++ ){
        for ( int x = 0; x < img.cols; x++ ){
            if ( max < img.at<uchar>( y, x ) ){
                max = img.at<uchar>( y, x );
            }
            if ( min > img.at<uchar>( y, x ) ){
                min = img.at<uchar>( y, x );
            }
        }
    }
    for ( int y = 0; y < img.rows; y++ ){
        for ( int x = 0; x < img.cols; x++ ){
            numImg.at<double>( y, x ) = img.at<uchar>( y, x ) * ( max - min ) / 255.0;
        }
    }
 
    //勾配方向
    Mat gradientOrientation = Mat::zeros( numImg.rows, numImg.cols, CV_64F );
    //勾配強度
    Mat gradientMagnitude = Mat::zeros( numImg.rows, numImg.cols, CV_64F );
    for ( int y = 0; y < numImg.rows; y++ ){
        for ( int x = 0; x < numImg.cols; x++ ){
            int x1 = 0;
            int x2 = 0;
            if ( x == 0 ){
                x1 = x;
                x2 = x + 1;
            }else if ( x == numImg.cols -1 ){
                x1 = x - 1;
                x2 = x;
            }else{
                x1 = x - 1;
                x2 = x + 1;
            }
 
            int y1 = 0;
            int y2 = 0;
            if ( y == 0 ){
                y1 = y;
                y2 = y + 1;
            }else if ( y == numImg.rows -1 ){
                y1 = y - 1;
                y2 = y;
            }else{
                y1 = y - 1;
                y2 = y + 1;
            }
            double fx = numImg.at<double>( y, x2 ) - numImg.at<double>( y, x1 );
            double fy = numImg.at<double>( y2, x ) - numImg.at<double>( y1, x );
 
            gradientMagnitude.at<double>( y, x ) = sqrt( fx * fx + fy * fy );
            if ( fx != 0 ){
                gradientOrientation.at<double>( y, x ) = atan( fy / fx );
            }else{
                gradientOrientation.at<double>( y, x ) = PI / 2;
            }
            if ( gradientOrientation.at<double>( y, x ) < 0 ){
                gradientOrientation.at<double>( y, x ) += PI;
            }
        }
    }
     
    //ヒストグラムを初期化する
    vector< vector< vector<double > > > histogram( numImg.rows / CELL_SIZE[1]);
    for ( int i = 0; i < numImg.rows / CELL_SIZE[1]; i++ ){
        histogram[i].resize( numImg.cols / CELL_SIZE[0]);
        for ( int j = 0; j < numImg.cols / CELL_SIZE[0]; j++ ){
            histogram[i][j].resize( GRADIENT_SIZE );
            for ( int k = 0; k < GRADIENT_SIZE; k++ ){
                histogram[i][j][k] = 0;
            }
        }
    }
 
    //ヒストグラムを作成する
    for ( int i = 0; i < numImg.rows / CELL_SIZE[1]; i++ ){
        for ( int j = 0; j < numImg.cols / CELL_SIZE[0]; j++ ){
            for ( int k = 0; k < CELL_SIZE[1]; k++ ){
                for ( int l = 0; l < CELL_SIZE[0]; l++ ){
                    int y = i * CELL_SIZE[1] + k;
                    int x = j * CELL_SIZE[0] + l;
                    int m = (int)(gradientOrientation.at<double>( y, x ) * 180 / PI / ( 180 / GRADIENT_SIZE ) );
                    histogram[i][j][m] += gradientMagnitude.at<double>( y, x );
                }
            }
        }
    }
 
    //ブロック領域での正規化
    Mat blockSum = Mat::zeros(numImg.rows / CELL_SIZE[1] - ( BLOCK_SIZE[1] - 1), numImg.cols / CELL_SIZE[0] - ( BLOCK_SIZE[0] - 1 ), CV_64F );
    for ( int i = 0; i < numImg.rows / CELL_SIZE[1] - ( BLOCK_SIZE[1] - 1); i++ ){
        for ( int j = 0; j < numImg.cols / CELL_SIZE[0] - ( BLOCK_SIZE[0] - 1 ); j++ ){
            for ( int k = 0; k < BLOCK_SIZE[1]; k++ ){
                for ( int l = 0; l < BLOCK_SIZE[0]; l++ ){
                    for ( int m = 0; m < GRADIENT_SIZE; m++ ){
                        blockSum.at<double>( i, j ) += pow( histogram[i+k][j+l][m], 2 );
                    }
                }
            }
        }
    }
    vector< vector< vector<double> > > histogram2 = histogram;
    for ( int i = 0; i < numImg.rows / CELL_SIZE[1] - ( BLOCK_SIZE[1] - 1); i++ ){
        for ( int j = 0; j < numImg.cols / CELL_SIZE[0] - ( BLOCK_SIZE[0] - 1 ); j++ ){
            for ( int k = 0; k < BLOCK_SIZE[1]; k++ ){
                for ( int l = 0; l < BLOCK_SIZE[0]; l++ ){
                    for ( int m = 0; m < GRADIENT_SIZE; m++ ){
                        histogram[i+k][j+l][m] = histogram2[i+k][j+l][m] / sqrt( blockSum.at<double>( i, j ) + 1 );
                    }
                }
            }
        }
    }
     
 
    //描画
    for ( int i = 0; i < numImg.rows / CELL_SIZE[1]; i++ ){
        line(img, Point( 0, i * CELL_SIZE[1] ), Point( numImg.cols, i * CELL_SIZE[1] ), Scalar(0,0,200), 1, 4);
    }
    for ( int i = 0; i < numImg.cols / CELL_SIZE[0]; i++ ){
        line(img, Point( i * CELL_SIZE[0], 0 ), Point( i * CELL_SIZE[0], numImg.rows ), Scalar(0,0,200), 1, 4);
    }
    for ( int i = 0; i < numImg.rows / CELL_SIZE[1]; i++ ){
        for ( int j = 0; j < numImg.cols / CELL_SIZE[0]; j++ ){
            for ( int k = 0; k < GRADIENT_SIZE; k++ ){
                if ( histogram[i][j][k] > 0.2 ){
                    int sy = i * CELL_SIZE[1] + 8*sin( PI * ( (180/GRADIENT_SIZE) * k + 90 ) / 180 ) +  CELL_SIZE[1] / 2;
                    int sx = j * CELL_SIZE[0] + 8*cos( PI * ( (180/GRADIENT_SIZE) * k + 90 ) / 180 ) + CELL_SIZE[0] / 2;
                    int dy = i * CELL_SIZE[1] - 8*sin( PI * ( (180/GRADIENT_SIZE) * k + 90 ) / 180 ) + CELL_SIZE[1] / 2;
                    int dx = j * CELL_SIZE[0] - 8*cos( PI * ( (180/GRADIENT_SIZE) * k + 90 ) / 180 ) + CELL_SIZE[0] / 2;
                    line(img, Point( sx, sy ), Point( dx, dy), Scalar(0,0,200), 1, 4);
                }
            }
        }
    }
 
    //保存
    imwrite( "hog4.jpg", img );
 
    return 0;
}
