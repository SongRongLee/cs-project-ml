#include"KnnBayesSemi.h"

KnnBayesSemi::KnnBayesSemi(vector<MyData> &X, vector<MyData> &XT, int k) {
	this->X = X;
	this->XT = XT;
	this->k = k;
	round_limit = 20;
	//combine data
	total_data = X;
	total_data.insert(total_data.end(), XT.begin(), XT.end());

	//set initial knn_label and class_weight
	for (int i = 0; i < total_data.size(); i++) {
		total_data[i].knn_label = total_data[i].label;
		total_data[i].class_w = 1;
	}

	preTrain();
}

void KnnBayesSemi::preTrain() {
	cout << "Pre-training..." << endl;
	KnnBayesTransD transd(X, XT, k);
	transd.performTrans(dis_matrixs, knn_results);
	for (int i = X.size(); i < X.size() + XT.size(); i++) {
		total_data[i].is_train = true;
	}
	cout << "Pre-train done." << endl;
}

void KnnBayesSemi::fillDismatrix() {
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

void KnnBayesSemi::performTrans() {
	double v = 0.1;
	vector<MyData> train_data(total_data.begin(), total_data.begin() + X.size() + XT.size());
	KNNClassifier knn(train_data, k);
	KNNClassifier first_knn(X, k);

	for (int rc = 0; rc < dis_matrixs.size() - 1; rc++) {
		double lambda = 1, epsilon;
		double r = 0.5;
		double w = 1.05;
		cout << "Round " << rc + 1 << " ." << endl;

		if (rc == 0) {
			//get knn class weight and label for testing data
			for (int i = train_data.size(); i < total_data.size(); i++) {
				vector<double> dis_vector(dis_matrixs[rc][i].begin(), dis_matrixs[rc][i].begin() + X.size());
				total_data[i].knn_label = first_knn.bayesprediction(total_data[i], dis_vector);
			}
		}
		else {
			//set knn label for XT
			for (int i = X.size(); i < train_data.size(); i++) {
				total_data[i].class_w_table = knn_results[rc][i - X.size()];
				int max_label;
				double max = -1;
				//find knn label
				for (int j = 0; j < total_data[j].class_w_table.size(); j++){
					if (total_data[i].class_w_table[j].second > max){
						max = total_data[i].class_w_table[j].second;
						max_label = total_data[i].class_w_table[j].first;
					}
				}
				total_data[i].knn_label = max_label;
			}
			//get knn class weight and label for testing data
			for (int i = train_data.size(); i < total_data.size(); i++) {
				vector<double> dis_vector(dis_matrixs[rc][i].begin(), dis_matrixs[rc][i].begin() + train_data.size());
				total_data[i].knn_label = knn.bayesprediction(total_data[i], dis_vector);
			}
		}
		
		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {
				if (i < train_data.size() && j < train_data.size()) {
					continue;
				}
				double f = 1;
				epsilon = lambda * total_data[i].class_w * total_data[j].class_w;
				if (r <= epsilon) {
					//change dis					
					f = 2 / (1 + exp(-w*v));
					if (total_data[i].knn_label == total_data[j].knn_label) {
						f = 1 / f;
					}
				}
				dis_matrixs[rc + 1][i][j] = dis_matrixs[rc][i][j] * f;
				dis_matrixs[rc + 1][j][i] = dis_matrixs[rc][i][j] * f;
			}
		}
	}
	printDismatrix(dis_matrixs[2]);
}