#ifndef KNNBAYSETRANSD
#define KNNBAYSETRANSD

#include "TransD.h"


class KnnBayesTransD : TransD
{
private:
	int k;
	int round_limit;
	vector<vector<double>> dis_matrix;
	vector<MyData> X;
	vector<MyData> T;
	vector<MyData> total_data;

public:
	KnnBayesTransD(vector<MyData> &X, vector<MyData> &T, int k);
	void performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<vector<vector<pair<int, double>>>> &knn_results);
};

#endif