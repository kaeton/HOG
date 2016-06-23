#include </usr/local/include/opencv2/core/core.hpp>
#include </usr/local/include/opencv2/imgproc/imgproc.hpp>
#include </usr/local/include/opencv2/highgui/highgui.hpp>
 
using namespace std;
using namespace cv;
 
/////////////////////////////////////
//グローバル変数
////////////////////////////////////
//円周率
const double PI = 3.141592;
// 8px * 8px
const vector<int> CELL_SIZE[2] = {8, 8};
// 2cell * 2cell
const vector<int> BLOCK_SIZE[2] = { 2, 2 };
//0度 - 180度 (20度ごと)
const int GRADIENT_SIZE = 9;
 
 
/////////////////////////////////////
//関数
////////////////////////////////////
 
/////////////////////////////////////
//名前：hogDesc
//引数１：img：HOG特徴量を抽出する画像
//引数２：hog：取り出したHOG特徴量を格納する
//戻り値：なし
//説明：画像imgからHOG特徴量を抽出し、hogにHOG特徴量を格納する。
//  ブロックサイズ等の変更はグローバル変数で指定する。
////////////////////////////////////
void hogDesc( Mat& img, vector<double>& hog ){
    //画像をvector型に変換する
    vector<vector <double> > numImg( img.rows, vector<double>( img.cols, 0 ) );
    for ( int y = 0; y < img.rows; y++ ){
        for ( int x = 0; x < img.cols; x++ ){
            numImg[y][x] = img.at<uchar>( y, x );
        }
    }
 
    //勾配方向
    vector<vector <double> > gradientOrientation( img.rows, vector<double>( img.cols, 0 ) );
    //勾配強度
    vector<vector <double> > gradientMagnitude( img.rows, vector<double>( img.cols, 0 ) );
    for ( int y = 0; y < img.rows; y++ ){
        for ( int x = 0; x < img.cols; x++ ){
            int x1 = 0;
            int x2 = 0;
            //画像の外側に出る場合で場合分け
            if ( x == 0 ){
                x1 = x;
                x2 = x + 1;
            }else if ( x == img.cols -1 ){
                x1 = x - 1;
                x2 = x;
            }else{
                x1 = x - 1;
                x2 = x + 1;
            }
 
            int y1 = 0;
            int y2 = 0;
            //画像の外側に出る場合で場合分け
            if ( y == 0 ){
                y1 = y;
                y2 = y + 1;
            }else if ( y == img.rows -1 ){
                y1 = y - 1;
                y2 = y;
            }else{
                y1 = y - 1;
                y2 = y + 1;
            }
 
            double fx = numImg[y][x2] - numImg[y][x1];
            double fy = numImg[y2][x] - numImg[y1][x];
 
            //勾配強度を出す
            gradientMagnitude[y][x] = sqrt( fx * fx + fy * fy );
            //勾配方向を出す。fxが0だと計算できないので小さい値を足しておく
            gradientOrientation[y][x] = atan2( fy , (fx+0.01) );
            //atan2の戻り値は-pi~piなので、値の範囲を0~piに直す。
            if ( gradientOrientation[y][x] < 0 ){
                gradientOrientation[y][x] = PI + gradientOrientation[y][x];
            }
        }
    }
     
    //ヒストグラムを初期化する
    vector< vector< vector<double > > > histogram( img.rows / CELL_SIZE[1]);
    for ( int i = 0; i < img.rows / CELL_SIZE[1]; i++ ){
        histogram[i].resize( img.cols / CELL_SIZE[0]);
        for ( int j = 0; j < img.cols / CELL_SIZE[0]; j++ ){
            histogram[i][j].resize( GRADIENT_SIZE );
            for ( int k = 0; k < GRADIENT_SIZE; k++ ){
                histogram[i][j][k] = 0;
            }
        }
    }
 
    //ヒストグラムを作成する
    for ( int i = 0; i < img.rows / CELL_SIZE[1]; i++ ){
        for ( int j = 0; j < img.cols / CELL_SIZE[0]; j++ ){
            for ( int k = 0; k < CELL_SIZE[1]; k++ ){
                for ( int l = 0; l < CELL_SIZE[0]; l++ ){
                    //画素の位置
                    int y = i * CELL_SIZE[1] + k;
                    int x = j * CELL_SIZE[0] + l;
                    //該当する２つの角度に投票する。その際角度の近さに応じて重みをつける。
                    int m1 = (int)((gradientOrientation[y][x] * 180 / PI) / ( 180 / GRADIENT_SIZE ) );
                    int m2 = (int)((gradientOrientation[y][x] * 180 / PI+(180/GRADIENT_SIZE)) / ( 180 / GRADIENT_SIZE ) );
                    double linInt = ((gradientOrientation[y][x] * 180 / PI)-(int)(gradientOrientation[y][x] * 180 / PI))/(double)(180/GRADIENT_SIZE);
                    if ( linInt == 0){
                        m2 = m1;
                    }
                    //１８０度付近のものは０度として投票する
                    if ( m1 >= histogram[i][j].size() ){
                        m1 = 0;
                    }
                    if ( m2 >= histogram[i][j].size() ){
                        m2 = 0;
                    }
                    //ヒストグラムに投票する
                    histogram[i][j][m1] += (1-linInt)*gradientMagnitude[y][x];
                    histogram[i][j][m2] += linInt*gradientMagnitude[y][x];
                }
            }
        }
    }
 
    //ブロック領域での正規化
    vector<vector <double> > blockSum( img.rows / CELL_SIZE[1] - ( BLOCK_SIZE[1] - 1), vector<double>(img.cols / CELL_SIZE[0] - ( BLOCK_SIZE[0] - 1 ), 0 ) );
    for ( int i = 0; i < img.rows / CELL_SIZE[1] - ( BLOCK_SIZE[1] - 1); i++ ){
        for ( int j = 0; j < img.cols / CELL_SIZE[0] - ( BLOCK_SIZE[0] - 1 ); j++ ){
            for ( int k = 0; k < BLOCK_SIZE[1]; k++ ){
                for ( int l = 0; l < BLOCK_SIZE[0]; l++ ){
                    for ( int m = 0; m < GRADIENT_SIZE; m++ ){
                        blockSum[i][j] += pow( histogram[i+k][j+l][m], 2 );
                    }
                }
            }
        }
    }
 
    //正規化回数
    int normalizeNum = ( img.rows / CELL_SIZE[1] - ( BLOCK_SIZE[1] - 1) ) * ( img.cols / CELL_SIZE[0] - ( BLOCK_SIZE[0] - 1 ) );
    //ブロック数
    int blockNum = BLOCK_SIZE[0] * BLOCK_SIZE[1];
    //HOG特徴量
    vector<double> result( normalizeNum * blockNum * GRADIENT_SIZE );
 
    //HOG特徴量を計算する
    int count = 0;
    for ( int i = 0; i < img.rows / CELL_SIZE[1] - ( BLOCK_SIZE[1] - 1); i++ ){
        for ( int j = 0; j < img.cols / CELL_SIZE[0] - ( BLOCK_SIZE[0] - 1 ); j++ ){
            for ( int k = 0; k < BLOCK_SIZE[1]; k++ ){
                for ( int l = 0; l < BLOCK_SIZE[0]; l++ ){
                    for ( int m = 0; m < GRADIENT_SIZE; m++ ){
                        result[count] = histogram[i+k][j+l][m] / sqrt( blockSum[i][j] + 1 );
                        if ( result[count] > 0.2 ){
                            result[count] = 0.2;
                        }
                        count++;
                    }
                }
            }
        }
    }
 
    hog = result;
}
 
/////////////////////////////////////
//名前：main
//引数：なし
//戻り値：あり
//説明：main関数
////////////////////////////////////
int main(){
    //画像
    Mat srcImg = imread( "pic01.jpg", CV_LOAD_IMAGE_GRAYSCALE );
     
    //HOG特徴量を取り出す
    vector<double> hog;
    hogDesc( srcImg, hog );
 
    return 0;
}
