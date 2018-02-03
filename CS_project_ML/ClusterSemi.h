#ifndef CLUSTERSEMI_H
#define CLUSTERSEMI_H

#include"SemiTransD.h"
#include"KnnBayesTransD.h"

class ClusterSemi : public SemiTransD
{
private:
	int k;
	int round_limit;
	vector<vector<vector<double>>> dis_matrixs;
	vector<int> knn_results;
	vector<MyData> X;
	vector<MyData> XT;
	vector<MyData> T;
	vector<MyData> total_data;
	Eigen::MatrixXd god_matrix;
	vector<vector<double>> dis_matrix;
	string folder;
	void erase(int i);
	void preTrain();
	void fillDismatrix();
	bool enablePrintLabel;

public:
	ClusterSemi(vector<MyData> &X, vector<MyData> &XT, int k, string folder, bool enablePrintLabel);
	void performTrans();
	void performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<int> &knn_results);
	void setT(vector<MyData> &T);
	void getSortedMatrix(vector<vector<double>> &new_dis, int i);
	void printSortedMatrixs();
	void printMatrixs(string folder);
	double getScore();
};

#endif