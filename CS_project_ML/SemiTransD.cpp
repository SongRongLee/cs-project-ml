#include"SemiTransD.h"

SemiTransD::SemiTransD() {
	k = 1;
	round_limit = 20;
}
SemiTransD::SemiTransD(vector<MyData> &X, vector<MyData> &XT, int k) {	
	this->X = X;
	this->XT = XT;
	this->k = k;
	round_limit = 20;
	//combine data
	total_data = X;
	total_data.insert(total_data.end(), XT.begin(), XT.end());

	//set initial knn_label and class_weight
	for (int i = 0; i < total_data.size(); i++) {
		if (i < this->X.size()) {
			this->X[i].knn_label = this->X[i].label;
			this->X[i].class_w = 1;
		}
		else {
			this->XT[i - this->X.size()].knn_label = this->XT[i - this->X.size()].label;
			this->XT[i - this->X.size()].class_w = 1;
		}
		total_data[i].knn_label = total_data[i].label;
		total_data[i].class_w = 1;
	}
	
	preTrain();
}
void SemiTransD::setK(int k) {
	this->k = k;
}
void SemiTransD::setRoundLimit(int round_limit) {
	this->round_limit = round_limit;
}
void SemiTransD::setT(vector<MyData> &T) {
	//reset previous data
	total_data.erase(total_data.begin() + X.size() + XT.size(), total_data.end());
	//set new data
	this->T = T;
	total_data.insert(total_data.end(), T.begin(), T.end());	
	fillDismatrix();
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
void SemiTransD::fillDismatrix() {
	//resize to X+XT+T square
	for (int i = 0; i < dis_matrixs.size(); i++) {
		dis_matrixs[i].resize(total_data.size());
		for (int j = 0; j < dis_matrixs[i].size(); j++) {
			dis_matrixs[i][j].resize(total_data.size());
		}
	}
	//initialize starting matrix
	genDismatrix(total_data, dis_matrixs[0]);
	printDismatrix(dis_matrixs[2]);
}

void SemiTransD::calNearList(vector<vector<int>> &near_list, int round) {	
	for (int i = 0; i < total_data.size(); i++) {
		vector<int> tempv;
		vector<pair<int, double>> sort_temp;
		for (int j = 0; j < total_data.size(); j++) {
			sort_temp.push_back(make_pair(j, dis_matrixs[round][i][j]));
		}
		sort(sort_temp.begin(), sort_temp.end(), mycomp);
		for (int j = 0; j < total_data.size(); j++) {
			tempv.push_back(sort_temp[j].first);
		}
		near_list.push_back(tempv);
	}
}
double SemiTransD::calw(int a, int b, vector<vector<int>> &near_list, int round) {
	double gamma = 0.01;

	double radius = dis_matrixs[round][a][b];
	double wij = 0, tmp_ratio;

	//calculate wij base on a's neighbors
	for (int i = 0; i < near_list[a].size(); i++) {
		int p = near_list[a][i];
		if (a != p && b != p) {
			if (dis_matrixs[round][a][p] > radius)break;
			if (dis_matrixs[round][a][p] > dis_matrixs[round][b][p])continue;
			tmp_ratio = dis_matrixs[round][b][p] / dis_matrixs[round][a][p];
			double A = tmp_ratio - 1;
			wij += (1 - exp(-gamma*A*A));
		}
	}

	//calculate wij base on b's neighbors
	for (int i = 0; i < near_list[b].size(); i++) {
		int p = near_list[b][i];
		if (a != p && b != p) {
			if (dis_matrixs[round][b][p] > radius)break;
			if (dis_matrixs[round][b][p] > dis_matrixs[round][a][p])continue;
			tmp_ratio = dis_matrixs[round][a][p] / dis_matrixs[round][b][p];
			double A = tmp_ratio - 1;
			wij += (1 - exp(-gamma*A*A));
		}
	}
	return wij;
}
void SemiTransD::performTrans() {
	double v = 0.1;
	int train_data_size = X.size() + XT.size();
	
	KNNClassifier first_knn(X, k);

	vector<vector<int>> near_list;

	for (int rc = 0; rc < dis_matrixs.size() - 1; rc++) {
		double lambda, epsilon;
		double r = 0.5;
		double w;		

		cout << "Round " << rc + 1 << " ." << endl;

		calNearList(near_list,rc);

		if (rc == 0) {
			//get knn class weight and label for testing data
			for (int i = train_data_size; i < total_data.size(); i++) {
				vector<double> dis_vector(dis_matrixs[rc][i].begin(), dis_matrixs[rc][i].begin() + X.size());
				total_data[i].knn_label = first_knn.prediction(total_data[i], dis_vector);
			}
		}
		else {
			//set knn label for XT
			for (int i = X.size(); i < train_data_size; i++) {
				total_data[i].knn_label = knn_results[rc][i - X.size()];
			}
			vector<MyData> train_data(total_data.begin(), total_data.begin() + X.size() + XT.size());
			KNNClassifier knn(train_data, k);
			//get knn class weight and label for testing data
			for (int i = train_data.size(); i < total_data.size(); i++) {
				vector<double> dis_vector(dis_matrixs[rc][i].begin(), dis_matrixs[rc][i].begin() + train_data.size());
				total_data[i].knn_label = knn.prediction(total_data[i], dis_vector);
			}
		}

		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {
				if (i < train_data_size && j < train_data_size) {
					continue;
				}
				double f = 1;
				if (total_data[i].is_train || total_data[j].is_train) {
					lambda = 0.5;
				}
				else {
					//both are labled data
					lambda = 1;
				}
				epsilon = lambda * total_data[i].class_w * total_data[j].class_w;
				if (r <= epsilon) {
					//change dis					
					w = calw(i, j, near_list, rc);
					f = 2 / (1 + exp(-w*v));
					if (total_data[i].knn_label == total_data[j].knn_label) {
						f = 1 / f;
					}
				}
				dis_matrixs[rc + 1][i][j] = dis_matrixs[rc][i][j] * f;
				dis_matrixs[rc + 1][j][i] = dis_matrixs[rc][i][j] * f;
			}
		}
		near_list.clear();
	}
	printDismatrix(dis_matrixs[2]);
}

void SemiTransD::getSortedMatrix(vector<vector<double>> &new_dis) {
	indexSortedMatrix(total_data, dis_matrixs[dis_matrixs.size()-1], new_dis);
}