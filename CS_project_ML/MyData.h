#ifndef MY_DATA_H
#define MY_DATA_H

#include<vector>;
using namespace std;
class MyData
{
public:
	vector<double> features;
	int num, label;
public:
	MyData();
	MyData(int num, vector<double> features, int label);
};

#endif 
