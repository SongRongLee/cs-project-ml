#include"KNNClassifier.h"

KNNClassifier::KNNClassifier() {}
KNNClassifier::KNNClassifier(vector<vector<float>> X, vector<int> Y):BaseClassifier(X,Y) {}

int KNNClassifier::prediction(vector<float> t) {

	int vsize = X.size();
	if (vsize == 0) {
		cout << "Prediciton error, not enough data.\n";
		return -1;
	}

	float min_dis = euDistance(X[0], t), temp_dis;
	int min_index = 0;	

	for (int i = 1; i < vsize; i++) {
		temp_dis = euDistance(X[i], t);
		if (temp_dis < min_dis) {
			min_dis = temp_dis;
			min_index = i;
		}
	}

	return Y[min_index];
}

vector<int> KNNClassifier::prediction(vector<vector<float>> TX) {

	int vsize = TX.size();
	vector<int> result;

	if (vsize == 0) {
		cout << "Prediciton error, not enough data.\n";
		return result;
	}	
	
	for (int i = 0; i < vsize; i++) {		
		result.push_back(prediction(TX[i]));
	}
	
	return result;
}

float KNNClassifier::euDistance(vector<float> a, vector<float> b) {

	if (a.size() != b.size()) {
		cout << "Euclidean distance error, size mismatch.\n";
		return -1;
	}

	int vsize = a.size();
	float tempsquare = 0;
	for (int i = 0; i < vsize; i++) {
		tempsquare += pow(a[i] - b[i], 2);
	}
	return sqrt(tempsquare);
}