#ifndef NMICLASSIFIER_H
#define NMICLASSIFIER_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"BaseClassifier.h"
#include"MyData.h"

class NMIClassifier : public BaseClassifier
{
private:
	int k;
public:
	vector<pair<int, double>> medoid;
	vector<pair<int, int >> medoid_idx;
	NMIClassifier();
	NMIClassifier(vector<MyData> X, int k);
	int prediction(MyData t);
	vector<int> prediction(vector<MyData> T);
	void setK(int k);
	void compute_medoid();
private:
	double euDistance(MyData a, MyData b);
};

#endif