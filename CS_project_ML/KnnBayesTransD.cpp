#include"KnnBayesTransD.h"

KnnBayesTransD::KnnBayesTransD(vector<MyData> &X, vector<MyData> &T, int k) {
	this->k = k;
	this->X = X;
	this->T = T;
	round_limit = 20;
	//generate distance matrix
	total_data = X;
	total_data.insert(total_data.end(), T.begin(), T.end());
	genDismatrix(total_data, dis_matrix);
	//set initial knn_label and class_weight
	for (int i = 0; i < total_data.size(); i++) {
		if (i >= X.size()) {
			total_data[i].knn_label = total_data[i].label;
			total_data[i].class_w = 1;
			total_data[i].class_w_table.push_back(pair<int, double>(0, 0));
		}
	}
}

void KnnBayesTransD::performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<vector<vector<pair<int, double>>>> &knn_results) {

	double v = 0.1;
	
	KNNClassifier one_nn(X, 1);
	NMIClassifier one_mi(X, dis_matrix, 1);

	vector<vector<double>> tmpdis;

	dis_matrixs.push_back(dis_matrix);
	tmpdis = dis_matrix;

	for (int rc = 0; rc < round_limit; rc++) {
		double lambda = 1, epsilon;
		double r = 0.5;
		double w = 1.05;

		cout << "Round " << rc + 1 << " ." << endl;

		//get knn class weight and label
		vector<vector<pair<int, double>>> tmpknn_result;
		KNNClassifier knn(total_data, k);
		for (int i = X.size(); i < total_data.size(); i++) {			
			vector<double> dis_vector(dis_matrix[i].begin(), dis_matrix[i].end());
			total_data[i].knn_label = knn.bayesprediction(total_data[i], dis_vector);
			tmpknn_result.push_back(total_data[i].class_w_table);
			//cout << "No." << total_data[i].num << " classify as " << total_data[i].knn_label << endl;
		}

		//record knn results
		knn_results.push_back(tmpknn_result);

		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {
				double f = 1;
				epsilon = lambda * total_data[i].class_w * total_data[j].class_w;
				if (r <= epsilon) {
					//change dis
					f = 1.05;
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

		//output class weight
		string title = "training_label" + to_string(rc + 1) + ".txt";
		ofstream out(title);
		for (int i = 0; i < total_data.size(); i++) {
			sort(total_data[i].class_w_table.begin(), total_data[i].class_w_table.end(), mycomp2);
			for (int j = 0; j < total_data[i].class_w_table.size(); j++) {
				out << fixed << setprecision(6) << total_data[i].class_w_table[j].first << "," << total_data[i].class_w_table[j].second << "\t";
			}
			out << endl;
		}
		out.close();	

		if (check_flag) {
			cout << "KnnBayesTransD done by 1-NN and 1mi match." << endl;
			break;
		}		
	}
}
