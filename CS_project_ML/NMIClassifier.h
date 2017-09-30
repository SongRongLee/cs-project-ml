#ifndef NMICLASSIFIER_H
#define NMICLASSIFIER_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"BaseClassifier.h"
#include"MyData.h"
#include"Utility.h"

class NMIClassifier : public BaseClassifier
{
private:
	int k;
	class Medoid
	{
	public:
		int label;
		int index;
		double min_dis;
	public:
		Medoid(int label, double min_dis, int index);
	};
public:
	vector<Medoid> medoids;

	NMIClassifier();
	NMIClassifier(vector<MyData> &X, int k);

	int prediction(MyData &t);
	vector<int> prediction(vector<MyData> &T);
	void setK(int k);
	void compute_medoid();
private:
	
};

#endif