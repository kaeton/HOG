#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "opencv_lib.h"


#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct _HOG_FEATURE{
    int bX;             //ブロック領域の左上の x 座標
    int bY;             //ブロック領域の左上の y 座標
    int cX;             //セル領域の左上の x 座標
    int cY;             //セル領域の左上の y 座標
    int ori;    //勾配の方向
    double dVal;        //特徴量
}HOG_FEATURE;

IplImage* Img;              //入力画像
IplImage* LowImg;           //リサイズ後の入力画像    
#define CELL_SIZE   5       //セルサイズ (pixel)
#define BLOCK_SIZE  3       //ブロックサイズ (セル)
#define SET_Y_SIZE  40      //パッチの縦幅 (pixel)
#define SET_X_SIZE  30      //パッチの横幅 (pixel)

#define ORIENTATION 9       //勾配方向

#define PI          3.14
#define LN_E        1.0


//HOG 特徴量の書き出し
int OutputHOGFeature(int nFeatureNum, HOG_FEATURE *pHOGFeatures, char* File){

    FILE *fp;
    int  i;

    if(NULL == (fp=fopen(File, "w"))){
        printf("Can not open the outfile.\n");
        return(1);  
    }

    for(i=0; i<nFeatureNum; i++){
        fprintf(fp, "%lf\n", pHOGFeatures[i].dVal);
    }

    fclose(fp);

    return(0);
}


//配列確保のためにあらかじめ特徴量の数を算出
int CountHOGFeature(){

    int i,j,k,x,y;
    int fCount = 0;     //特徴量の数

    //ブロックの移動
    for(y=0; y<((SET_Y_SIZE/CELL_SIZE)-BLOCK_SIZE+1); y++){
        for(x=0; x<((SET_X_SIZE/CELL_SIZE)-BLOCK_SIZE+1); x++){     
            //セル内の移動
            for(j=0; j<BLOCK_SIZE; j++){
                for(i=0; i<BLOCK_SIZE; i++){
                    //勾配方向数
                    for(k=0; k<ORIENTATION; k++){
                        //特徴量数のカウント
                        fCount++;
                    }
                }
            }
        }
    }

    return (fCount);
}

//勾配方向ヒストグラムの算出
int CompHistogram(double cell_hist[]){

    int x,y;
    int num1,num2;      //配列の要素数の計算

    int xDelta,yDelta;          //各方向の輝度の差分
    double magnitude;           //勾配強度

    double angle = 180.0/(double)ORIENTATION;   //量子化する時の角度
    double gradient;

    //画像を格納している構造体から輝度値だけコピー
    unsigned char* imgSource = (unsigned char*)LowImg->imageData;
    //画像を格納している構造体の水平方向のデータ数
    int wStep = LowImg->widthStep;
    //1ピクセルあたりのビット数
    int bpp   = ((LowImg->depth&255)/8)*LowImg->nChannels;

    //パッチ内の移動
    for(y=0; y<SET_Y_SIZE; y++){
        for(x=0; x<SET_X_SIZE; x++){

            //横方向の差分
            if(x == 0){
                num1 = y*wStep+(x+0)*bpp;
                num2 = y*wStep+(x+1)*bpp;
                xDelta = imgSource[num1]-imgSource[num2];
            }else if(x == LowImg->width-1){
                num1 = y*wStep+(x-1)*bpp;
                num2 = y*wStep+(x+0)*bpp;
                xDelta = imgSource[num1]-imgSource[num2];
            }else{
                num1 = y*wStep+(x-1)*bpp;
                num2 = y*wStep+(x+1)*bpp;
                xDelta = imgSource[num1]-imgSource[num2];
            }

            //縦方向の差分
            if(y == 0){
                num1 = (y+0)*wStep+x*bpp;
                num2 = (y+1)*wStep+x*bpp;
                yDelta = imgSource[num1]-imgSource[num2];
            }else if(y == LowImg->height-1){
                num1 = (y-1)*wStep+x*bpp;
                num2 = (y+0)*wStep+x*bpp;
                yDelta = imgSource[num1]-imgSource[num2];
            }else{
                num1 = (y-1)*wStep+x*bpp;
                num2 = (y+1)*wStep+x*bpp;
                yDelta = imgSource[num1]-imgSource[num2];
            }

            //勾配強度の算出
            magnitude = sqrt((double)xDelta*(double)xDelta+(double)yDelta*(double)yDelta);

            //勾配方向の算出
            gradient = atan2((double)yDelta, (double)xDelta);
            //ラジアンから角度へ変換
            gradient = (gradient*180.0)/PI;
            //符号が負である場合は反転
            if(gradient < 0.0){
                gradient += 360.0;
            }
            //0~360度から 0~180度に変換
            if(gradient > 180.0){
                gradient -= 180.0;
            }
            //20度ずつ,9分割
            gradient = gradient/angle;

            //ヒストグラムに蓄積
            num1 = (int)((y/CELL_SIZE)*(SET_X_SIZE/CELL_SIZE)*ORIENTATION+(x/CELL_SIZE)*ORIENTATION+(int)gradient);
            cell_hist[num1] += magnitude;
        }
    }
    return(0);
}

//HOG 特徴量の算出と正規化
int CompHOG(HOG_FEATURE *pHOGFeatures, double cell_hist[]){

    int x,y,i,j,k,f;
    int num;                //配列の要素数の計算

    double sum_magnitude;   //ブロック領域内の
    int    fCount = 0;      //特徴量数のカウント

    //特徴量のパラメータを算出
    //ブロックの移動
    for(y=0; y<((SET_Y_SIZE/CELL_SIZE)-BLOCK_SIZE+1); y++){
        for(x=0; x<((SET_X_SIZE/CELL_SIZE)-BLOCK_SIZE+1); x++){ 
            //セル内の移動
            for(j=0; j<BLOCK_SIZE; j++){
                for(i=0; i<BLOCK_SIZE; i++){
                    //勾配方向¸
                    for(k=0; k<ORIENTATION; k++){
                        //特徴量のパラメータ
                        pHOGFeatures[fCount].bX = x;
                        pHOGFeatures[fCount].bY = y;
                        pHOGFeatures[fCount].cX = i;
                        pHOGFeatures[fCount].cY = j;
                        pHOGFeatures[fCount].ori = k;

                        //特徴量数のカウント
                        fCount++;
                    }
                }
            }
        }
    }

    //特徴量の算出と正規化
    for(f=0; f<fCount; f++){
        //特徴量量を正規化する時に使用する特徴量の二乗和を正規化
        sum_magnitude = 0.0;
        //セル内の移動
        for(j=0; j<BLOCK_SIZE; j++){
            for(i=0; i<BLOCK_SIZE; i++){

                //勾配方向
                for(k=0; k<ORIENTATION; k++){
                    //正規化のためヒストグラムの総和の二乗を算出
                    num = (int)((pHOGFeatures[f].bY+j)*(SET_X_SIZE/CELL_SIZE)*ORIENTATION+(pHOGFeatures[f].bX+i)*ORIENTATION+k);
                    sum_magnitude += cell_hist[num]*cell_hist[num];
                }
            }
        }

        //特徴量の正規化
        num = (int)((pHOGFeatures[f].bY+pHOGFeatures[f].cY)*(SET_X_SIZE/CELL_SIZE)*ORIENTATION+(pHOGFeatures[f].bX+pHOGFeatures[f].cX)*ORIENTATION+pHOGFeatures[f].ori);
        if(sum_magnitude==0) pHOGFeatures[f].dVal=0;
        else if(sum_magnitude!=0)pHOGFeatures[f].dVal = cell_hist[num] / sum_magnitude;
    }

    return (0);
}

//勾配強度と勾配方向を算出して HOG 特徴量を抽出
int HOG(HOG_FEATURE *pHOGFeatures){

    int    i,j,k;
    int    num;
    double *cell_hist;

    //勾配方向ヒストグラムを保存する領域を確保
    num = (SET_Y_SIZE/CELL_SIZE)*(SET_X_SIZE/CELL_SIZE)*ORIENTATION;
    cell_hist = new double[num];

    //0を代入して初期化
    for(j=0; j<(SET_Y_SIZE/CELL_SIZE); j++){
        for(i=0; i<(SET_X_SIZE/CELL_SIZE); i++){
            for(k=0; k<ORIENTATION; k++){
                num = j*(SET_X_SIZE/CELL_SIZE)*ORIENTATION+i*ORIENTATION+k;
                cell_hist[num] = 0.0;
            }
        }
    }

    //勾配方向と勾配強度から勾配方向ヒストグラムを算出
    CompHistogram(cell_hist);

    //HOG 特徴量の算出
    CompHOG(pHOGFeatures, cell_hist);

    //確保していた高米方向ヒストグラムの領域を解放
    delete []cell_hist;

    return (0);
}


int main(int argc, char** argv){

    int         nHOGFeatureNum;
    HOG_FEATURE *pHOGFeatures;
    char FileName[256];
    char buffer[256];

    //画像の読み込み
    if(NULL == (Img=cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE))){
        printf("Can not open the file.\n");
        return(1);
    }

    //画像のサイズを一定にするための構造体
    CvSize r_size = cvSize((int)SET_X_SIZE, (int)SET_Y_SIZE);
    LowImg        = cvCreateImage(r_size, IPL_DEPTH_8U, 1);

    //パッチサイズにリサイズ
    cvResize(Img, LowImg, CV_INTER_LINEAR);

    //あらかじめ特徴量の数を算出
    nHOGFeatureNum = CountHOGFeature();
    //扱う特徴量の数だけ構造体を確保
    pHOGFeatures = new HOG_FEATURE[nHOGFeatureNum];
    //HOG 特徴抽出
    HOG(pHOGFeatures);

    sscanf(argv[1],"%s.png",buffer);
    sprintf(FileName,"%s.hog",buffer);

    //特徴量の書き出し
    OutputHOGFeature(nHOGFeatureNum, pHOGFeatures, FileName);

    //画像のデータが入っている構造体を解放
    cvReleaseImage(&LowImg);
    cvReleaseImage(&Img);

    //特徴量の構造体を解放
    delete []pHOGFeatures;

    return (0);
}
