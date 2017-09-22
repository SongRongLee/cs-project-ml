#ifndef MY_DATA_H
#define MY_DATA_H

#include<vector>;
using namespace std;
class MyData
{
public:
	vector<float> features;
	int num, label;
public:
	MyData();
	MyData(int num, vector<float> features, int label);
};

#endif 
