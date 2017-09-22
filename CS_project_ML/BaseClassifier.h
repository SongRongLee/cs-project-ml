#ifndef BASE_CLASSIFIER_H
#define BASE_CLASSIFIER_H

#include<vector>
#include<string>
#include"MyData.h"

using namespace std;

class BaseClassifier
{
protected:
	int data_count;
	vector<MyData> X;
public:
	BaseClassifier();
	BaseClassifier(vector<MyData> X);
	void addData(MyData x);
	virtual int prediction(MyData t) = 0;
	virtual vector<int> prediction(vector<MyData> T) = 0;
};

#endif