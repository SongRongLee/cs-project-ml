#include"SemiTransD.h"

SemiTransD::SemiTransD() {
	k = 1;
	round_limit = 20;
}
SemiTransD::SemiTransD(vector<MyData> &X, vector<MyData> &XT, vector<MyData> &T, int k) {	
	this->X = X;
	this->XT = XT;
	this->T = T;
	this->k = k;
	round_limit = 20;
	//combine data
	total_data = X;
	total_data.insert(total_data.end(), XT.begin(), XT.end());
	total_data.insert(total_data.end(), T.begin(), T.end());
	//set initial knn_label and class_weight
	for (int i = 0; i < X.size(); i++) {
		total_data[i].knn_label = total_data[i].label;
		total_data[i].class_w = 1;
	}
	for (int i = X.size(); i < total_data.size(); i++) {
		total_data[i].class_w = 1;
	}
}

void SemiTransD::setK(int k) {
	this->k = k;
}
void SemiTransD::setRoundLimit(int round_limit) {
	this->round_limit = round_limit;
}
void SemiTransD::preTrain() {
	cout << "Pre-training..." << endl;
	TransD transd(X, XT, k);
	transd.performTrans(dis_matrixs, knn_results);
	for (int i = X.size(); i < X.size() + XT.size(); i++) {
		total_data[i].is_train = true;
	}
	cout << "Pre-train done." << endl;
}
void SemiTransD::performTrans(vector<vector<double>> &new_dis) {

	
}

void SemiTransD::getSortedMatrix(vector<vector<double>> &new_dis) {
	
}