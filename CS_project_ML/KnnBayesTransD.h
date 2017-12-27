#ifndef KNNBAYSETRANSD_H
#define KNNBAYSETRANSD_H
#define THREAD_NUM 8

#include "TransD.h"
#include <thread>

class KnnBayesTransD : public TransD
{
private:
	int k;
	int round_limit;
	vector<vector<double>> dis_matrix;
	vector<MyData> X;
	vector<MyData> T;
	vector<MyData> total_data;
	void predict_thread(int n);
public:
	KnnBayesTransD(vector<MyData> &X, vector<MyData> &T, int k);
	void performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<vector<vector<pair<int, double>>>> &knn_results);
};

#endif