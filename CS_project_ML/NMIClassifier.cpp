#include"NMIClassifier.h"

NMIClassifier::NMIClassifier() {
	k = 1;
}
NMIClassifier::NMIClassifier(vector<MyData> X, int k) :BaseClassifier(X) {
	this->k = k;
}

NMIClassifier::Medoid::Medoid(int label, double min_dis, int index = 0) {
	this->index = index;
	this->label = label;
	this->min_dis = min_dis;
}
void NMIClassifier::setK(int k) {
	this->k = k;
}

vector<int> NMIClassifier::prediction(vector<MyData> T) {

	int vsize = T.size();
	compute_medoid();
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
			if (i != j){
				if (X[i].label == X[j].label){
					dis_sum += euDistance(X[i], X[j]);
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

	//print medoids
	for (int i = 0; i < medoids.size(); i++) {
		cout << "medoid[" << i << "] is " << endl;
		cout << X[medoids[i].index];
	}
}
int NMIClassifier::prediction(MyData t){
	double min_dis = -1;
	int min_label = 0;
	/*for (int i = 0; i < medoid.size(); i++){
		cout << medoid[i].first << " " << medoid[i].second << " " << medoid_idx[i].second << endl;
	}*/
	for (int i = 0; i < medoids.size(); i++){
		double temp_dis = euDistance(X[medoids[i].index], t);
		if (temp_dis < min_dis || min_dis < 0){
			min_dis = temp_dis;
			min_label = medoids[i].label;
		}
	}
	return min_label;
}