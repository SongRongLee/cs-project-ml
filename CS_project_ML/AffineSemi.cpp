#include"AffineSemi.h"
AffineSemi::AffineSemi(vector<MyData> &X, vector<MyData> &XT, int k) {
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

void AffineSemi::preTrain() {
	//cout << "Pre-training..." << endl;
	KnnBayesTransD transd(X, XT, k);
	transd.performTrans(dis_matrixs, knn_results);
	for (int i = X.size(); i < X.size() + XT.size(); i++) {
		total_data[i].is_train = true;
	}

	//setup first matrix
	Eigen::MatrixXd first_matrix(total_data.size()+1, total_data.size());
	for (int i = 0; i < total_data.size(); i++) {
		vector<double> tmp_vector(dis_matrixs[0][i]);
		first_matrix.row(i) = Eigen::VectorXd::Map(&tmp_vector[0], tmp_vector.size());
	}
	vector<double> tmp_vector(total_data.size(), 1);
	first_matrix.row(total_data.size()) = Eigen::VectorXd::Map(&tmp_vector[0], tmp_vector.size());
	
	//setup last matrix
	Eigen::MatrixXd last_matrix(total_data.size()+1, total_data.size());
	for (int i = 0; i < total_data.size(); i++) {
		vector<double> tmp_vector(dis_matrixs[dis_matrixs.size() - 1][i]);
		last_matrix.row(i) = Eigen::VectorXd::Map(&tmp_vector[0], tmp_vector.size());
	}
	tmp_vector = vector<double>(total_data.size(), 1);
	last_matrix.row(total_data.size()) = Eigen::VectorXd::Map(&tmp_vector[0], tmp_vector.size());

	//calculate god matrix
	Eigen::MatrixXd first_matrix_inverse = first_matrix.completeOrthogonalDecomposition().pseudoInverse();
	//Eigen::MatrixXd first_matrix_inverse = first_matrix.inverse();
	god_matrix = last_matrix*first_matrix_inverse;
	tmp_vector = vector<double>(total_data.size(), 0);
	tmp_vector.push_back(1);
	god_matrix.row(total_data.size()) = Eigen::VectorXd::Map(&tmp_vector[0], tmp_vector.size());

	/*ofstream f1out("last_matrix.txt");
	f1out << last_matrix;
	f1out.close();

	ofstream f2out("god_matrix.txt");
	f2out << god_matrix;
	f2out.close();

	cout << "Pre-train done." << endl;*/
}

void AffineSemi::setT(vector<MyData> &T) {
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

void AffineSemi::fillDismatrix() {
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

void AffineSemi::performTrans() {
	int train_data_size = X.size() + XT.size();

	for (int i = train_data_size; i < total_data.size(); i++)
	{
		vector<double> tmp_vector(dis_matrixs[0][i]);
		tmp_vector.push_back(1);
		Eigen::VectorXd test = Eigen::VectorXd::Map(&tmp_vector[0], train_data_size + 1);
		//Eigen::VectorXd test = Eigen::VectorXd::Map(&tmp_vector[0], train_data_size);
		Eigen::VectorXd final_dis = god_matrix*test;
		//set new dis
		for (int j = 0; j < train_data_size; j++)
		{
			dis_matrixs[dis_matrixs.size() - 1][i][j] = final_dis(j, 0);
		}
		//output new dis
		string title = "testing_dis_inverse" + to_string(T[i - train_data_size].num) + ".txt";
		ofstream out(title);
		out << final_dis;
		out.close();
	}
}
void AffineSemi::getSortedMatrix(vector<vector<double>> &new_dis) {
	indexSortedMatrix(total_data, dis_matrixs[dis_matrixs.size() - 1], new_dis);
}
double AffineSemi::getScore() {
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