#ifndef NONLINEARSEMI_H
#define NONLINEARSEMI_H

#include"SemiTransD.h"
#include"KnnBayesTransD.h"

class NonlinearSemi : public SemiTransD
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
	void preTrain();
	void fillDismatrix();

public:
	NonlinearSemi(vector<MyData> &X, vector<MyData> &XT, int k);
	void performTrans();
	void setT(vector<MyData> &T);
	void getSortedMatrix(vector<vector<double>> &new_dis);
	double getScore();
};

#endif