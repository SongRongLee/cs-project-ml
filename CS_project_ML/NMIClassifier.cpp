#include"NMIClassifier.h"

NMIClassifier::NMIClassifier() {
	k = 1;
}
NMIClassifier::NMIClassifier(vector<MyData> &X, int k) :BaseClassifier(X) {
	this->k = k;
	compute_medoid();
}

NMIClassifier::NMIClassifier(vector<MyData> &X, vector<vector<double>> &dis_matrix, int k) : BaseClassifier(X) {
	this->k = k;
	this->dis_matrix = dis_matrix;
	compute_medoid();
}

NMIClassifier::Medoid::Medoid(int label, double min_dis, int index = 0) {
	this->index = index;
	this->label = label;
	this->min_dis = min_dis;
}
void NMIClassifier::setK(int k) {
	this->k = k;
}
void NMIClassifier::setDisMatrix(vector<vector<double>> &dis_matrix) {
	this->dis_matrix = dis_matrix;
	medoids.clear();
	compute_medoid();
}
void NMIClassifier::compute_medoid(){

	//parsing all labels
	for (int i = 0; i < X.size(); i++){
		int flag = 0;
		for (int j = 0; j < medoids.size(); j++){
			if (X[i].label == medoids[j].label){
				flag = 1;
				break;
			}
		}
		if (flag == 0){
			medoids.push_back(Medoid(X[i].label, -1));
		}
	}

	//select medoids
	for (int i = 0; i < X.size(); i++){
		double dis_sum = 0;
		int label_idx;
		for (int j = 0; j < X.size(); j++){
			if (i != j && X[i].label == X[j].label){
				if (dis_matrix.empty()) {
					dis_sum += calDistance(X[i], X[j], dis_type);
				}
				else {
					dis_sum += dis_matrix[i][j];
				}
			}
		}
		for (int j = 0; j < medoids.size(); j++){
			if (X[i].label == medoids[j].label){
				label_idx = j;
				break;
			}
		}
		if (dis_sum < medoids[label_idx].min_dis || medoids[label_idx].min_dis < 0){
			medoids[label_idx].min_dis = dis_sum;
			medoids[label_idx].index = i;
		}
	}
}

vector<int> NMIClassifier::prediction(vector<MyData> &T) {

	int vsize = T.size();
	vector<int> result;

	if (vsize == 0) {
		cout << "Prediciton error, not enough data.\n";
		return result;
	}

	for (int i = 0; i < vsize; i++) {
		result.push_back(prediction(T[i]));
	}
	return result;
}

int NMIClassifier::prediction(MyData &t){
	double min_dis = -1;
	int min_label = 0;
	/*for (int i = 0; i < medoid.size(); i++){
		cout << medoid[i].first << " " << medoid[i].second << " " << medoid_idx[i].second << endl;
	}*/
	for (int i = 0; i < medoids.size(); i++){
		double temp_dis = calDistance(X[medoids[i].index], t, dis_type);
		if (temp_dis < min_dis || min_dis < 0){
			min_dis = temp_dis;
			min_label = medoids[i].label;
		}
	}
	return min_label;
}

void NMIClassifier::printMedoids() {
	for (int i = 0; i < medoids.size(); i++) {
		cout << "medoids[" << i << "] is " << endl;
		cout << X[medoids[i].index];
		cout << "with dis_sum = " << medoids[i].min_dis << endl << endl;;
	}
}