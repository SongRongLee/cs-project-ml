#include"ClusterSemi.h"
#include<omp.h>
ClusterSemi::ClusterSemi(vector<MyData> &X, vector<MyData> &XT, int k, string folder, bool enablePrintLabel) {

	this->X = X;
	this->XT = XT;
	this->k = k;
	this->T = XT;
	round_limit = 20;
	//combine data
	total_data = X;
	total_data.insert(total_data.end(), XT.begin(), XT.end());
	genDismatrix(total_data, dis_matrix);
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
	this->folder = folder;
	this->enablePrintLabel = enablePrintLabel;
	preTrain();
}

void ClusterSemi::preTrain() {
	//cout << "Pre-training..." << endl;

	performTrans(dis_matrixs, knn_results);
	for (int i = X.size(); i < X.size() + XT.size(); i++) {
		total_data[i].real_label = knn_results[i - X.size()];
	}

	Eigen::MatrixXd first_matrix(total_data.size(), total_data.size());
	for (int i = 0; i < total_data.size(); i++)
		first_matrix.row(i) = Eigen::VectorXd::Map(&dis_matrixs[0][i][0], dis_matrixs[0][i].size());
	Eigen::MatrixXd last_matrix(total_data.size(), total_data.size());
	for (int i = 0; i < total_data.size(); i++)
		last_matrix.row(i) = Eigen::VectorXd::Map(&dis_matrixs[dis_matrixs.size() - 1][i][0], dis_matrixs[dis_matrixs.size() - 1][i].size());
	//Eigen::MatrixXd first_matrix_inverse = first_matrix.completeOrthogonalDecomposition().pseudoInverse();
	Eigen::MatrixXd first_matrix_inverse = first_matrix.inverse();
	god_matrix = first_matrix_inverse*last_matrix;

	/*ofstream f2out("god_matrix.txt");
	f2out << god_matrix;
	f2out.close();*/

	//cout << "Pre-train done." << endl;
}

void ClusterSemi::setT(vector<MyData> &T) {
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

void ClusterSemi::fillDismatrix() {
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

void ClusterSemi::performTrans() {
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
void ClusterSemi::getSortedMatrix(vector<vector<double>> &new_dis,int i) {
	indexSortedMatrix(total_data, dis_matrixs[i], new_dis);
}
void ClusterSemi::printSortedMatrixs()
{
	string outstr = "matrix";
	for (int i = 0; i < dis_matrixs.size(); i++)
	{
		
		vector<vector<double>> sorted_dis_matrix;
		getSortedMatrix(sorted_dis_matrix,i);
		ofstream out(folder+outstr + to_string(i) + ".txt");
		for (int i = 0; i < sorted_dis_matrix.size(); i++) {
			for (int j = 0; j < sorted_dis_matrix[i].size(); j++) {
				out << left << fixed << setprecision(18) << setw(21) << sorted_dis_matrix[i][j]+1;
			}
			out << endl;
		}
	}
}
void ClusterSemi::printMatrixs(string folder)
{
	string outstr = "matrix";
	
	for (int i = 0; i < dis_matrixs.size(); i++)
	{

		vector<vector<double>> sorted_dis_matrix;
		ofstream out(folder + outstr + to_string(i) + ".txt");
		for (int i = 0; i < sorted_dis_matrix.size(); i++) {
			for (int j = 0; j < sorted_dis_matrix[i].size(); j++) {
				out << left << fixed << setprecision(18) << setw(21) << sorted_dis_matrix[i][j];
			}
			out << endl;
		}
		
	}
}
double ClusterSemi::getScore() {
	vector<int> results;
	int train_data_size = X.size() + XT.size();
	vector<MyData> train_data(total_data.begin(), total_data.begin() + train_data_size);
	//set real_label to label
	for (int j = 0; j < train_data.size(); j++) {
		train_data[j].label = train_data[j].real_label;
	}
	KNNClassifier adaptive_knn(train_data, 1);
	for (int i = train_data_size; i < total_data.size(); i++) {
		vector<double> dis_vector(dis_matrixs[dis_matrixs.size() - 1][i].begin(), dis_matrixs[dis_matrixs.size() - 1][i].begin() + train_data_size);
		//results.push_back(adaptive_knn.adaptive_prediction(total_data[i], dis_vector));
		results.push_back(adaptive_knn.prediction(total_data[i], dis_vector));
	}
	if (enablePrintLabel == true)
	{

		ofstream outknn(folder + "knn_included_test" + ".txt");
		ofstream outreal(folder + "real_included_test" + ".txt");
		printlabel(total_data, outknn, outreal, results);
	}
	double wrong_count = checkResult(results, T);
	return (double)(T.size() - wrong_count) / (double)T.size() * 100;
}
void ClusterSemi::erase(int i) {
	total_data.erase(total_data.begin() + i);
	XT.erase(XT.begin() + (i-X.size()));
	T.erase(T.begin() + (i - X.size()));

}
void ClusterSemi::performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<int> &knn_results) {

	double v = 0.1;

	KNNClassifier one_nn(X, 1);
	NMIClassifier one_mi(X, dis_matrix, 1);

	vector<vector<double>> tmpdis;

	dis_matrixs.push_back(dis_matrix);
	knn_results.resize(T.size());
	tmpdis = dis_matrix;

	for (int rc = 0; rc < round_limit; rc++) {
		double lambda = 1, epsilon;
		double r = 0.5;
		double w = 1.05;

		//cout << "Round " << rc + 1 << " ." << endl;

		//create thread
		vector<thread> threads;
		vector<vector<pair<int, double>>> tmpknn_result;
		KNNClassifier knn(total_data, k);
		//get knn class weight and label	
#pragma omp parallel for
		for (int i = X.size(); i < total_data.size(); i++) {
			vector<double> dis_vector(dis_matrix[i].begin(), dis_matrix[i].end());
			total_data[i].knn_label = knn.bayesprediction(total_data[i], dis_vector);
		}
		
		if (enablePrintLabel == true)
		{
			
			ofstream outknn(folder + "knn" + to_string(rc+1) + ".txt");
			ofstream outreal(folder + "real" + to_string(rc + 1) + ".txt");
			printlabel(total_data, outknn, outreal);
		}
			
		//set bayes knn results
		/*if (rc == 0)
		{
			for (int i = X.size(); i < total_data.size(); i++) {
				if (total_data[i].class_w < 0.6)
				{
					erase(i);
					i--;
				}
			}
			dis_matrixs.pop_back();
			genDismatrix(total_data, dis_matrix);
			dis_matrixs.push_back(dis_matrix);
			tmpdis = dis_matrix;
		}*/

		for (int i = X.size(); i < total_data.size(); i++) {
			knn_results[i - X.size()] = total_data[i].knn_label;
		}

		
		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {
				double f = 1;
				//
				/*if (total_data[i].is_train || total_data[j].is_train) {
				lambda = 1;
				}
				else {
				lambda = 0.5;
				}*/
				//
				epsilon = lambda * total_data[i].class_w * total_data[j].class_w;
				if (r <= epsilon) {
					if (lambda == 0.5) {
						cout << total_data[i].class_w << " " << total_data[j].class_w << endl;
					}
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
				//cout << "T[" << i << "] fail, 1nn = " << knn_result << ", 1mi = " << nmi_result << endl;
				break;
			}
		}

		//output class weight
		/*string title = "training_label" + to_string(rc + 1) + ".txt";
		ofstream out(title);
		double *beauty_weight = new double[total_data.back().class_w_table.size()];
		for (int j = 0; j < total_data.back().class_w_table.size(); j++)
		{
		beauty_weight[j] = 0;
		}
		vector<MyData>sorted_data = total_data;
		sort(sorted_data.begin(), sorted_data.end(), mycompindex);
		for (int i = 0; i < sorted_data.size(); i++) {
		sort(sorted_data[i].class_w_table.begin(), sorted_data[i].class_w_table.end(), mycomp2);
		for (int j = 0; j < sorted_data[i].class_w_table.size(); j++)
		{
		beauty_weight[sorted_data[i].class_w_table[j].first] = sorted_data[i].class_w_table[j].second;
		}
		for (int j = 0; j <sorted_data.back().class_w_table.size(); j++) {
		out << fixed << setprecision(6) << j << "," << beauty_weight[j] << "\t";
		beauty_weight[j] = 0;
		}
		out << endl;
		}
		out.close();
		*/
		if (check_flag) {
			//cout << "KnnBayesTransD done by 1-NN and 1mi match." << endl;
			break;
		}
		
	}

	
}