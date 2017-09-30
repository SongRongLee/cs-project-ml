#include"TransD.h"

TransD::TransD() {
	k = 1;
}
TransD::TransD(vector<MyData> &X, vector<MyData> &T, int k){
	this->k = k;
	this->X = X;
	this->T = T;

	//generate distance matrix
	vector<MyData> total_data = X;
	total_data.insert(total_data.end(), T.begin(), T.end());
	genDismatrix(total_data,dis_matrix);

	//set initial knn_label and class_weight
	for (int i = 0; i < X.size(); i++) {
		X[i].knn_label = X[i].label;
		X[i].class_w = 1;
	}
	for (int i = 0; i < T.size(); i++) {
		T[i].knn_label = T[i].label;
		T[i].class_w = 1;
	}
}

void TransD::setK(int k) {
	this->k = k;
}

void TransD::performTrans(vector<vector<double>> &new_dis) {
	KNNClassifier knn(X, k);
	for (int i = 0; i < T.size(); i++) {
		vector<double> dis_vector(dis_matrix[X.size() + i].begin(), dis_matrix[X.size() + i].begin() + X.size());
		T[i].knn_label = knn.prediction(T[i], dis_vector);
	}
	//to do...

}