// 15-8-31-testbow.cpp : 定义控制台应用程序的入口点。


#include "stdlib.h"
#include "stdio.h"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <windows.h>
#define image_num 75
using namespace cv;
using namespace std;
/***************************************************************
* 特征检测类型参数设置
*****************************************************************/
struct FDMParams
{
	string featureDetectorType; //特征检测器类型
	string descriptorType; //特征描述子类型
	string matcherType; //特征匹配方法类型
	FDMParams() :featureDetectorType("SIFT"), descriptorType("SIFT"), matcherType("BruteForce")
	{
	}
	FDMParams(string _featureDetectorType, string _descriptorType, string _matcherType)
	{
		featureDetectorType = _featureDetectorType;
		descriptorType = _descriptorType;
		matcherType = _matcherType;
	}
	void printMessage()
	{
		cout << "feature detector type is : " << featureDetectorType << endl;
		cout << "descriptor extractor type is: " << descriptorType << endl;
		cout << "matcher type is : " << matcherType << endl<<endl;
	}
};

/********************************************************************
* 从指定文件读取K均值生成的字典
*********************************************************************/
bool readVocabulary(string& vocFileName, Mat& vocabulary)
{
	cout << "Reading vocabulary...";
	FileStorage fs(vocFileName, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["vocabulary"] >> vocabulary;
		cout << "Vocabulary read over. " << endl << endl;
		return true;
	}
	return false;
}

int main(int argc, const char *  argv [])
{
	std::string image_path;
	image_path = std::string(argv[1]);
	Mat tfIdfOfBase;//训练集中每幅图片的tf_idf
    cout<<"读入训练集的tf_idf向量集..."<<endl;
    FileStorage fs(".\\save\\wordfre.xml", FileStorage::READ);//读入训练集的tf_idf向量集
    fs["wordFreq"] >> tfIdfOfBase;
    fs.release();
    cout<<"读取完毕 "<<endl;
    
    cout<<"读入视觉单词表..."<<endl;
    Mat vocabulary;
    fs.open(".\\save\\vocabulary.xml", FileStorage::READ);//读入训练过程中构建的视觉单词表
	fs["vocabulary"] >> vocabulary;
    fs.release();
    cout<<"读取完毕"<<endl;

	int clusterNumber=vocabulary.rows;

	 //读入倒排索引------------------------------------------------------------------------------------------
    cout<<"读取倒排索引..."<<endl;
    vector<vector<int>> invertedList(clusterNumber);
    ifstream in;
    int value;
    in.open(".\\save\\SinvertedList.txt");
    for( int i = 0 ; i < invertedList.size(); i ++ ){
        in >>value;
        while(value != -1){
            
            invertedList[i].push_back(value);
            in >> value ;
        }
    }
    in.close();
    cout<<"读取完毕"<<endl;

	initModule_nonfree();  
	FDMParams fdmParams;
	fdmParams = FDMParams();//使用默认的特征提取方法"SIFT"
	fdmParams.printMessage();
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create(fdmParams.featureDetectorType);
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create(fdmParams.descriptorType);
	Ptr<DescriptorMatcher> descriptorMatch = DescriptorMatcher::create(fdmParams.matcherType);
	BOWImgDescriptorExtractor bowDE(new SiftDescriptorExtractor(),new FlannBasedMatcher());
	bowDE.setVocabulary(vocabulary);
	string test_image_path=".\\save\\test-image\\"+image_path;
	//string test_image_path=".\\save\\test-image\\12.jpg";
     Mat img1 = imread(test_image_path,1);
	imshow("查询图像",img1);
	vector<KeyPoint> kp;
	featureDetector ->detect(img1,kp);
	Mat descriptors;
	bowDE.compute(img1,kp,descriptors);

    cout<<"计算测试图片的TF-IDF..."<<endl;
    Mat idfMat ;//读入训练时候得到的各聚类中心的IDF
    fs.open(".\\save\\IDF.xml", FileStorage::READ);
    fs["IDF"] >> idfMat;
    fs.release();
	
    Mat tfIdfOfImg(1,clusterNumber,CV_32F);

    for(int i = 0 ; i < clusterNumber; i ++ ){//计算传入的图片的tf-idf向量
		//tfIdfOfImg.at<float>(0,i) = idfMat.at<float>(0,i)*descriptors.at<float>(0,i);
		tfIdfOfImg.at<float>(0,i) = descriptors.at<float>(0,i);
    }
    cout<<"计算完毕"<<endl;


	cout<<"开始相似图片匹配..."<<endl;
	 int k = -1;//用来记录相似度最高的图片的id号
	double similarity[image_num] = {0};//用来表征相似度
    int flag[image_num] = {0};
	int indices[image_num]={0};
    for(int i = 0 ; i < tfIdfOfImg.cols; i ++ ){
        
        if(tfIdfOfImg.at<float>(0,i) != 0){//若该图片内包含有第i个视觉单词
            vector<int> imgsId = invertedList[i];//获得倒排索引中，第i个聚类中心对应的所有图片的id
            for(int j = 0 ; j < imgsId.size(); j ++ ){//遍历倒排索引中包含该视觉单词的所有图片
                int id = imgsId[j];
                if(flag[id] == 0){
                    flag[id] = 1;//排除已经计算过的图片id
                    double vectorMultiply = 0.00;//词频向量向量积
                    double vector1Modulo = 0.00;//待检测图片词频向量向量积
                    double vector2Modulo = 0.00;//图库内图片词频向量向量积
                    
                    for( int t = 0 ; t < clusterNumber; t++){//计算余弦相似度
                        
                        vector1Modulo += 0.1 * tfIdfOfBase.ptr<float>(id)[t] * tfIdfOfBase.ptr<float>(id)[t];
                        vector2Modulo += 0.1 * tfIdfOfImg.at<float>(0,t) * tfIdfOfImg.at<float>(0,t);
						vectorMultiply += 0.1 * tfIdfOfBase.ptr<float>(id)[t] * tfIdfOfImg.at<float>(0,t)*idfMat.at<float>(0,t);
                        
                    }
                    
                    vector1Modulo = sqrt(vector1Modulo);
                    vector2Modulo = sqrt(vector2Modulo);
					similarity[id] = vectorMultiply/(vector1Modulo*vector2Modulo);
                }
            }
            
        }
    }
	//冒泡法排序
	for(size_t i=0;i<image_num;i++)
    {
        indices[i]=i;
    }

	int tmp_index; double tmp;

	for(int i=0;i<image_num;i++)
    {
		for(int j=i+1;j<image_num;j++)
        {
			if(similarity[i]<similarity[j])
            {
                tmp=similarity[i];
                similarity[i]=similarity[j];
                similarity[j]=tmp;

                tmp_index=indices[i];
                indices[i]=indices[j];
                indices[j]=tmp_index;
            }
        }
    }
    
   
/****
*返回最相似的图片
*/
	string trainFileName=".\\save\\train\\";
	string filter;
	 string path;
	 char str[10];
	 sprintf_s(str,"%2d",indices[0]);
	 path=trainFileName+"image_"+str+".jpg";
	  Mat imgshow=imread(path,1);//检索到的图片
	  imshow("返回结果",imgshow);
	 cvWaitKey(0);
	  return 0;
}
