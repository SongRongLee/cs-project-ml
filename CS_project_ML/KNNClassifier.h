#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"BaseClassifier.h"
#include"MyData.h"
#include"Utility.h"

class KNNClassifier : public BaseClassifier 
{
private:
	int k;
public:
	KNNClassifier();
	KNNClassifier(vector<MyData> X, int k);
	int prediction(MyData t);
	vector<int> prediction(vector<MyData> T);
	void setK(int k);
};

#endif