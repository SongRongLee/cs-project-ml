#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include"BaseClassifier.h"
#include<iostream>
#include<cmath>

class KNNClassifier : public BaseClassifier 
{
public:
	KNNClassifier();
	KNNClassifier(vector<vector<float>> X, vector<int> Y);
	int prediction(vector<float> x);
	vector<int> prediction(vector<vector<float>> TX);
private:
	float euDistance(vector<float> a, vector<float> b);
};

#endif