#ifndef CLUSTERSEMI_H
#define CLUSTERSEMI_H

#include"SemiTransD.h"
#include"KnnBayesTransD.h"
#include"KnnBayesSemi.h"
//wrapper 
void TransD(string prefix, string folder, string datalist[] );
class ClusterSemi : public SemiTransD
{
private:
	int k;
	int round_limit;
	vector<double> weights;
	vector<vector<vector<double>>> dis_matrixs;
	vector<int> knn_results;
	vector<MyData> X;
	vector<MyData> XT;
	vector<MyData> T;
	vector<MyData> total_data;
	Eigen::MatrixXd god_matrix;
	Eigen::MatrixXd cluster_matrix;
	vector<vector<double>> dis_matrix;
	vector<vector<double>> before_dis_matrix;
	string folder;
	void erase(int i);


	void performTrans(vector<vector<vector<double>>>& dis_matrixs, vector<int>& knn_results);
	void preTrain();
	void calNearList(vector<vector<int>>& near_list, vector<vector<double>> dis_matrix);
	double calw(int a, int b, vector<vector<int>>& near_list, vector<vector<double>> dis_matrix);
	vector<vector<double>> PreCluster(vector<vector<double>> clu_dis_matrix);
	void fillDismatrix();
	bool enablePrintLabel;

public:
	ClusterSemi(vector<MyData> &X, vector<MyData> &XT, int k, string folder, bool enablePrintLabel);
	void performTrans();
	void setT(vector<MyData> &T);
	void getSortedMatrix(vector<vector<double>> &new_dis, int i);
	void printSortedMatrixs();
	void printMatrixs(string folder);
	double getScore();
};

#endif