#ifndef BAYESKNNCLASSIFIER_H
#define BAYESKNNCLASSIFIER_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"BaseClassifier.h"
#include"MyData.h"
#include"Utility.h"

class bayesKNNClassifier : public BaseClassifier
{
public:
	bayesKNNClassifier();
	bayesKNNClassifier(vector<MyData> &X, int k);
	int prediction(MyData &t);
	int prediction(MyData &t, vector<double> dis_vector);
	vector<int> prediction(vector<MyData> &T);
};



#endif
