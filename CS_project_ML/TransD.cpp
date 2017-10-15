#include"TransD.h"

bool mycomp(pair<int, double> a, pair<int, double> b) {
	return a.second < b.second;
}

TransD::TransD() {
	k = 1;
	round_limit = 20;
}
TransD::TransD(vector<MyData> &X, vector<MyData> &T, int k){
	this->k = k;
	this->X = X;
	this->T = T;
	round_limit = 20;
	//generate distance matrix
	total_data = X;
	total_data.insert(total_data.end(), T.begin(), T.end());
	genDismatrix(total_data, dis_matrix);
	//set initial knn_label and class_weight
	for (int i = 0; i < X.size(); i++) {
		total_data[i].knn_label = total_data[i].label;
		total_data[i].class_w = 1;
	}
	for (int i = X.size(); i < total_data.size(); i++) {
		total_data[i].class_w = 1;
	}
}

void TransD::setK(int k) {
	this->k = k;
}
void TransD::setRoundLimit(int round_limit) {
	this->round_limit = round_limit;
}
void TransD::calNearList(vector<vector<int>> &near_list) {
	for (int i = 0; i < total_data.size(); i++) {
		vector<int> tempv;
		vector<pair<int, double>> sort_temp;
		for (int j = 0; j < total_data.size(); j++) {
			sort_temp.push_back(make_pair(j, dis_matrix[i][j]));
		}
		sort(sort_temp.begin(), sort_temp.end(), mycomp);
		for (int j = 0; j < total_data.size(); j++) {
			tempv.push_back(sort_temp[j].first);
		}
		near_list.push_back(tempv);
	}
}
double TransD::calw(int a, int b, vector<vector<int>> &near_list) {
	double gamma = 0.01;

	double radius = dis_matrix[a][b];	
	double wij = 0, tmp_ratio;

	//calculate wij base on a's neighbors
	for (int i = 0; i < near_list[a].size(); i++) {
		int p = near_list[a][i];
		if (a != p && b != p) {
			if (dis_matrix[a][p] > radius)break;
			if (dis_matrix[a][p] > dis_matrix[b][p])continue;
			tmp_ratio = dis_matrix[b][p] / dis_matrix[a][p];
			double A = tmp_ratio - 1;
			wij += (1 - exp(-gamma*A*A));
		}
	}

	//calculate wij base on b's neighbors
	for (int i = 0; i < near_list[b].size(); i++) {
		int p = near_list[b][i];
		if (a != p && b != p) {
			if (dis_matrix[b][p] > radius)break;
			if (dis_matrix[b][p] > dis_matrix[a][p])continue;
			tmp_ratio = dis_matrix[a][p] / dis_matrix[b][p];
			double A = tmp_ratio - 1;
			wij += (1 - exp(-gamma*A*A));
		}
	}

	return wij;
}
void TransD::performTrans(vector<vector<double>> &new_dis) {

	double v = 0.1;

	KNNClassifier knn(X, k);
	KNNClassifier one_nn(X, 1);
	NMIClassifier one_mi(X, dis_matrix, 1);

	new_dis = dis_matrix;
	vector<vector<int>> near_list;

	for (int rc = 0; rc < round_limit; rc++) {
		double lambda, epsilon;
		double r = 0.5;
		double w;
		cout << "Round " << rc+1 << " ." << endl;

		calNearList(near_list);
		//get knn class weight and label
		for (int i = X.size(); i < total_data.size(); i++) {
			vector<double> dis_vector(dis_matrix[i].begin(), dis_matrix[i].begin() + X.size());
			total_data[i].knn_label = knn.prediction(total_data[i], dis_vector);
			//cout << "No." << total_data[i].num << " classify as " << total_data[i].knn_label << endl;
		}

		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {				
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
					w = calw(i, j, near_list);
					f = 2 / (1 + exp(-w*v));
					if (total_data[i].knn_label == total_data[j].knn_label) {
						f = 1 / f;
					}
				}
				new_dis[i][j] = dis_matrix[i][j] * f;
				new_dis[j][i] = new_dis[i][j];
			}
		}
		dis_matrix = new_dis;
		near_list.clear();

		//verify 1-nn and 1mi
		int knn_result, nmi_result;
		bool check_flag = true;

		one_mi.setDisMatrix(dis_matrix);
		//cout << "1-nn  nmi" << endl;
		for (int i = 0; i < T.size(); i++) {
			vector<double> dis_vector(dis_matrix[X.size() + i].begin(), dis_matrix[X.size() + i].begin() + X.size());
			knn_result = one_nn.prediction(T[i], dis_vector);
			nmi_result = one_mi.prediction(T[i], dis_vector);
			//cout << knn_result << "    " << nmi_result << endl;
			if (knn_result != nmi_result) {
				check_flag = false;
				cout << "T[" << i << "] fail, 1nn = " << knn_result << ", 1mi = " << nmi_result << endl;
				break;
			}
		}
		
		if (check_flag) {
			cout << "TransD done by 1-NN and 1mi match." << endl;
			break;
		}
	}
}

void TransD::performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<vector<int>> &knn_results) {

	double v = 0.1;

	KNNClassifier knn(X, k);
	KNNClassifier one_nn(X, 1);
	NMIClassifier one_mi(X, dis_matrix, 1);
	
	vector<vector<int>> near_list;
	vector<vector<double>> tmpdis;

	dis_matrixs.push_back(dis_matrix);
	tmpdis = dis_matrix;

	for (int rc = 0; rc < round_limit; rc++) {
		double lambda, epsilon;
		double r = 0.5;
		double w;
		
		cout << "Round " << rc + 1 << " ." << endl;

		calNearList(near_list);

		//get knn class weight and label
		vector<int> tmpknn_result;
		for (int i = X.size(); i < total_data.size(); i++) {
			vector<double> dis_vector(dis_matrix[i].begin(), dis_matrix[i].begin() + X.size());
			total_data[i].knn_label = knn.prediction(total_data[i], dis_vector);
			tmpknn_result.push_back(total_data[i].knn_label);
			//cout << "No." << total_data[i].num << " classify as " << total_data[i].knn_label << endl;
		}
		//record knn results
		knn_results.push_back(tmpknn_result);
		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {
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
					w = calw(i, j, near_list);
					f = 2 / (1 + exp(-w*v));
					if (total_data[i].knn_label == total_data[j].knn_label) {
						f = 1 / f;
					}
				}
				tmpdis[i][j] = dis_matrix[i][j] * f;
				tmpdis[j][i] = tmpdis[i][j];
			}
		}
		dis_matrix = tmpdis;
		//record distance matrixs
		dis_matrixs.push_back(dis_matrix);
		near_list.clear();

		//verify 1-nn and 1mi
		int knn_result, nmi_result;
		bool check_flag = true;

		one_mi.setDisMatrix(dis_matrix);
		//cout << "1-nn  nmi" << endl;
		for (int i = 0; i < T.size(); i++) {
			vector<double> dis_vector(dis_matrix[X.size() + i].begin(), dis_matrix[X.size() + i].begin() + X.size());
			knn_result = one_nn.prediction(T[i], dis_vector);
			nmi_result = one_mi.prediction(T[i], dis_vector);
			//cout << knn_result << "    " << nmi_result << endl;
			if (knn_result != nmi_result) {
				check_flag = false;
				cout << "T[" << i << "] fail, 1nn = " << knn_result << ", 1mi = " << nmi_result << endl;
				break;
			}
		}

		if (check_flag) {
			cout << "TransD done by 1-NN and 1mi match." << endl;
			break;
		}
	}
}

void TransD::getSortedMatrix(vector<vector<double>> &new_dis) {
	indexSortedMatrix(total_data, dis_matrix, new_dis);
}