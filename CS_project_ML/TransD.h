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
	int round_limit;
	vector<vector<double>> dis_matrix;
	vector<MyData> X;
	vector<MyData> T;
	vector<MyData> total_data;
	double calw(int a, int b, vector<vector<int>> &near_list);
	void calNearList(vector<vector<int>> &near_list);
public:
	TransD();
	TransD(vector<MyData> &X, vector<MyData> &T, int k);
	void setK(int k);	
	void setRoundLimit(int round_limit);
	void performTrans(vector<vector<double>> &new_dis);
	void performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<vector<int>> &knn_results);
	void getSortedMatrix(vector<vector<double>> &new_dis);
};

#endif