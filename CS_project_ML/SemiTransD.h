#ifndef SEMITRANSD_H
#define SEMITRANSD_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"MyData.h"
#include"Utility.h"
#include"TransD.h"

class SemiTransD
{
private:
	int k;
	int round_limit;
	vector<vector<vector<double>>> dis_matrixs;
	vector<vector<int>> knn_results;
	vector<MyData> X;
	vector<MyData> XT;
	vector<MyData> T;
	vector<MyData> total_data;
public:
	SemiTransD();
	SemiTransD(vector<MyData> &X, vector<MyData> &XT, vector<MyData> &T, int k);
	void setK(int k);
	void setRoundLimit(int round_limit);
	void preTrain();
	void performTrans(vector<vector<double>> &new_dis);
	void getSortedMatrix(vector<vector<double>> &new_dis);
};

#endif