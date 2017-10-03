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
	class Medoid
	{
	public:
		int label;
		int index;
		double min_dis;
	public:
		Medoid(int label, double min_dis, int index);
	};

	int k;
	vector<Medoid> medoids;
	vector<vector<double>> dis_matrix;
public:	
	NMIClassifier();
	NMIClassifier(vector<MyData> &X, int k);
	NMIClassifier(vector<MyData> &X, vector<vector<double>> &dis_matrix, int k);

	int prediction(MyData &t);
	vector<int> prediction(vector<MyData> &T);
	void setK(int k);	
	void printMedoids();
	void setDisMatrix(vector<vector<double>> &dis_matrix);

private:
	void compute_medoid();
};

#endif