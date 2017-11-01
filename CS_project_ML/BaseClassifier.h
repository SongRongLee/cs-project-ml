#ifndef BASE_CLASSIFIER_H
#define BASE_CLASSIFIER_H

#include<vector>
#include<string>
#include"MyData.h"

using namespace std;

class BaseClassifier
{
protected:
	int dis_type;
	vector<MyData> X;
public:
	BaseClassifier();
	BaseClassifier(vector<MyData> &X);
	void addData(MyData x);
	void BaseClassifier::setDistype(int dis_type);
	virtual int prediction(MyData &t) = 0;
	virtual vector<int> prediction(vector<MyData> &T) = 0;
};

#endif