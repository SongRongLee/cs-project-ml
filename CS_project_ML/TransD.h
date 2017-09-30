#ifndef TRANSD_H
#define TRANSD_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"MyData.h"
#include"Utility.h"
#include"KNNClassifier.h"
#include"NMIClassifier.h"

class TransD
{
private:
	int k;
	vector<vector<double>> dis_matrix;
	vector<MyData> X;
	vector<MyData> T;
public:
	TransD();
	TransD(vector<MyData> &X, vector<MyData> &T, int k);
	void setK(int k);
	void performTrans(vector<vector<double>> &new_dis);
};

#endif