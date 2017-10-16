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

	void preTrain();
	void fillDismatrix();
	double calw(int a, int b, vector<vector<int>> &near_list, int round);
	void calNearList(vector<vector<int>> &near_list, int round);

public:
	SemiTransD();
	SemiTransD(vector<MyData> &X, vector<MyData> &XT, int k);
	void setK(int k);
	void setT(vector<MyData> &T);
	void setRoundLimit(int round_limit);
	
	void performTrans();
	void getSortedMatrix(vector<vector<double>> &new_dis);
};

#endif