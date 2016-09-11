// 15-8-31-testbow.cpp : �������̨Ӧ�ó������ڵ㡣


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
* ����������Ͳ�������
*****************************************************************/
struct FDMParams
{
	string featureDetectorType; //�������������
	string descriptorType; //��������������
	string matcherType; //����ƥ�䷽������
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
* ��ָ���ļ���ȡK��ֵ���ɵ��ֵ�
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
	Mat tfIdfOfBase;//ѵ������ÿ��ͼƬ��tf_idf
    cout<<"����ѵ������tf_idf������..."<<endl;
    FileStorage fs(".\\save\\wordfre.xml", FileStorage::READ);//����ѵ������tf_idf������
    fs["wordFreq"] >> tfIdfOfBase;
    fs.release();
    cout<<"��ȡ��� "<<endl;
    
    cout<<"�����Ӿ����ʱ�..."<<endl;
    Mat vocabulary;
    fs.open(".\\save\\vocabulary.xml", FileStorage::READ);//����ѵ�������й������Ӿ����ʱ�
	fs["vocabulary"] >> vocabulary;
    fs.release();
    cout<<"��ȡ���"<<endl;

	int clusterNumber=vocabulary.rows;

	 //���뵹������------------------------------------------------------------------------------------------
    cout<<"��ȡ��������..."<<endl;
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
    cout<<"��ȡ���"<<endl;

	initModule_nonfree();  
	FDMParams fdmParams;
	fdmParams = FDMParams();//ʹ��Ĭ�ϵ�������ȡ����"SIFT"
	fdmParams.printMessage();
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create(fdmParams.featureDetectorType);
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create(fdmParams.descriptorType);
	Ptr<DescriptorMatcher> descriptorMatch = DescriptorMatcher::create(fdmParams.matcherType);
	BOWImgDescriptorExtractor bowDE(new SiftDescriptorExtractor(),new FlannBasedMatcher());
	bowDE.setVocabulary(vocabulary);
	string test_image_path=".\\save\\test-image\\"+image_path;
	//string test_image_path=".\\save\\test-image\\12.jpg";
     Mat img1 = imread(test_image_path,1);
	imshow("��ѯͼ��",img1);
	vector<KeyPoint> kp;
	featureDetector ->detect(img1,kp);
	Mat descriptors;
	bowDE.compute(img1,kp,descriptors);

    cout<<"�������ͼƬ��TF-IDF..."<<endl;
    Mat idfMat ;//����ѵ��ʱ��õ��ĸ��������ĵ�IDF
    fs.open(".\\save\\IDF.xml", FileStorage::READ);
    fs["IDF"] >> idfMat;
    fs.release();
	
    Mat tfIdfOfImg(1,clusterNumber,CV_32F);

    for(int i = 0 ; i < clusterNumber; i ++ ){//���㴫���ͼƬ��tf-idf����
		//tfIdfOfImg.at<float>(0,i) = idfMat.at<float>(0,i)*descriptors.at<float>(0,i);
		tfIdfOfImg.at<float>(0,i) = descriptors.at<float>(0,i);
    }
    cout<<"�������"<<endl;


	cout<<"��ʼ����ͼƬƥ��..."<<endl;
	 int k = -1;//������¼���ƶ���ߵ�ͼƬ��id��
	double similarity[image_num] = {0};//�����������ƶ�
    int flag[image_num] = {0};
	int indices[image_num]={0};
    for(int i = 0 ; i < tfIdfOfImg.cols; i ++ ){
        
        if(tfIdfOfImg.at<float>(0,i) != 0){//����ͼƬ�ڰ����е�i���Ӿ�����
            vector<int> imgsId = invertedList[i];//��õ��������У���i���������Ķ�Ӧ������ͼƬ��id
            for(int j = 0 ; j < imgsId.size(); j ++ ){//�������������а������Ӿ����ʵ�����ͼƬ
                int id = imgsId[j];
                if(flag[id] == 0){
                    flag[id] = 1;//�ų��Ѿ��������ͼƬid
                    double vectorMultiply = 0.00;//��Ƶ����������
                    double vector1Modulo = 0.00;//�����ͼƬ��Ƶ����������
                    double vector2Modulo = 0.00;//ͼ����ͼƬ��Ƶ����������
                    
                    for( int t = 0 ; t < clusterNumber; t++){//�����������ƶ�
                        
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
	//ð�ݷ�����
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
*���������Ƶ�ͼƬ
*/
	string trainFileName=".\\save\\train\\";
	string filter;
	 string path;
	 char str[10];
	 sprintf_s(str,"%2d",indices[0]);
	 path=trainFileName+"image_"+str+".jpg";
	  Mat imgshow=imread(path,1);//��������ͼƬ
	  imshow("���ؽ��",imgshow);
	 cvWaitKey(0);
	  return 0;
}
