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
		if (i < this->X.size()) {
			this->X[i].knn_label = this->X[i].label;
			this->X[i].class_w = 1;
			this->X[i].class_w_table.push_back(pair<int, double>(this->X[i].label, 1));
			total_data[i].knn_label = total_data[i].label;
			total_data[i].class_w = 1;
			total_data[i].class_w_table.push_back(pair<int, double>(total_data[i].label, 1));
		}
		else {
			total_data[i].knn_label = total_data[i].label;
			total_data[i].class_w = 0;
			total_data[i].class_w_table.push_back(pair<int, double>(0, 0));
		}		
	}

	preTrain();
}

void KnnBayesSemi::preTrain() {
	//cout << "Pre-training..." << endl;
	KnnBayesTransD transd(X, XT, k);
	transd.performTrans(dis_matrixs, knn_results);
	for (int i = X.size(); i < X.size() + XT.size(); i++) {
		total_data[i].is_train = true;
	}

	Eigen::MatrixXd first_matrix(total_data.size(), total_data.size());
	for (int i = 0; i < total_data.size(); i++)
		first_matrix.row(i) = Eigen::VectorXd::Map(&dis_matrixs[0][i][0], dis_matrixs[0][i].size());
	Eigen::MatrixXd last_matrix(total_data.size(), total_data.size());
	for (int i = 0; i < total_data.size(); i++)
		last_matrix.row(i) = Eigen::VectorXd::Map(&dis_matrixs[dis_matrixs.size()-1][i][0], dis_matrixs[dis_matrixs.size() - 1][i].size());
	//Eigen::MatrixXd first_matrix_inverse = first_matrix.completeOrthogonalDecomposition().pseudoInverse();
	Eigen::MatrixXd first_matrix_inverse = first_matrix.inverse();
	god_matrix = first_matrix_inverse*last_matrix;

	/*ofstream f2out("god_matrix.txt");
	f2out << god_matrix;
	f2out.close();*/

	//cout << "Pre-train done." << endl;
}

void KnnBayesSemi::setT(vector<MyData> &T) {
	//reset previous data
	total_data.erase(total_data.begin() + X.size() + XT.size(), total_data.end());
	//set new data
	this->T = T;
	total_data.insert(total_data.end(), T.begin(), T.end());
	for (int i = X.size() + XT.size(); i < total_data.size(); i++) {
		total_data[i].knn_label = total_data[i].label;
		total_data[i].class_w = 0;
		total_data[i].class_w_table.push_back(pair<int, double>(0, 0));
	}
	fillDismatrix();
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
}

void KnnBayesSemi::performTrans() {
	int train_data_size = X.size() + XT.size();
	/*for (int rc = 0; rc < dis_matrixs.size() - 1; rc++) {
		double lambda = 1, epsilon;
		double r = 0.5;
		
		cout << "Round " << rc + 1 << " ." << endl;
		
		//get knn class weight and label for testing data
		for (int i = train_data_size; i < total_data.size(); i++) {
			vector<MyData> train_data(total_data.begin(), total_data.begin() + X.size() + XT.size());
			train_data.push_back(total_data[i]);
			KNNClassifier knn(train_data, k);
			vector<double> dis_vector(dis_matrixs[rc][i].begin(), dis_matrixs[rc][i].begin() + train_data_size);
			dis_vector.push_back(0);
			total_data[i].knn_label = knn.bayesprediction(total_data[i], dis_vector);
		}

		//set knn label for XT
		for (int i = X.size(); i < train_data_size; i++) {
			total_data[i].class_w_table = knn_results[rc][i - X.size()];
			int max_label;
			double max = -1;
			//find knn label
			for (int j = 0; j < total_data[i].class_w_table.size(); j++){
				if (total_data[i].class_w_table[j].second > max){
					max = total_data[i].class_w_table[j].second;
					max_label = total_data[i].class_w_table[j].first;
				}
			}				
			total_data[i].knn_label = max_label;
			total_data[i].class_w = max;
		}				
		
		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {
				if (i < train_data_size && j < train_data_size) {
					continue;
				}
				double f = 1;
				epsilon = lambda * total_data[i].class_w * total_data[j].class_w;
				if (r <= epsilon) {
					//change dis					
					f = 1.05;
					if (total_data[i].knn_label == total_data[j].knn_label) {
						f = 1 / f;
					}
				}
				dis_matrixs[rc + 1][i][j] = dis_matrixs[rc][i][j] * f;
				dis_matrixs[rc + 1][j][i] = dis_matrixs[rc][i][j] * f;
			}
		}
		

	}*/
	/*
	//print dismatrix		
	string title = "testing_dis" + to_string(dis_matrixs.size()) + ".txt";
	ofstream out(title);
	vector<vector<double>> new_dis;
	indexSortedMatrix(total_data, dis_matrixs[dis_matrixs.size() - 1], new_dis);
	printDismatrix(new_dis, out);
	out.close();*/
	
	
	for (int i = train_data_size; i < total_data.size(); i++)
	{
		Eigen::VectorXd test = Eigen::VectorXd::Map(&dis_matrixs[0][i][0], train_data_size);
		Eigen::VectorXd final_dis = test.transpose()*god_matrix;
		//set new dis
		for (int j = 0; j < train_data_size; j++)
		{
			dis_matrixs[dis_matrixs.size() - 1][i][j] = final_dis(j, 0);
		}
		//output new dis
		/*string title = "testing_dis_inverse" + to_string(T[i- train_data_size].num) + ".txt";
		ofstream out(title);
		out << final_dis;
		out.close();*/
	}
	//maybe wrong
	/*for (int i = 0; i < T.size(); i++)
	{
		string title = "testing_dis" + to_string(T[i].num )+ ".txt";
		ofstream out(title);
		vector<vector<vector<double>>> new_diss;
		//indexSortedAllMatrix(total_data, dis_matrixs, new_diss);
		printTestDis(dis_matrixs, T[i].num, total_data, out);
		out.close();
	}*/
	

}
void KnnBayesSemi::getSortedMatrix(vector<vector<double>> &new_dis) {
	indexSortedMatrix(total_data, dis_matrixs[dis_matrixs.size() - 1], new_dis);
}
double KnnBayesSemi::getScore() {
	vector<int> results;
	int train_data_size = X.size() + XT.size();
	vector<MyData> train_data(total_data.begin(), total_data.begin() + train_data_size);
	//set real_label to label
	for (int j = 0; j < train_data.size(); j++) {
		train_data[j].label = train_data[j].real_label;
	}
	KNNClassifier one_nn(train_data, 1);
	for (int i = train_data_size; i < total_data.size(); i++) {		
		vector<double> dis_vector(dis_matrixs[dis_matrixs.size() - 1][i].begin(), dis_matrixs[dis_matrixs.size() - 1][i].begin() + train_data_size);
		results.push_back(one_nn.prediction(total_data[i], dis_vector));
	}
	double wrong_count = checkResult(results, T);
	return (double)(T.size() - wrong_count) / (double)T.size() * 100;
}