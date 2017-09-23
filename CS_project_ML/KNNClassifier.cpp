#include"KNNClassifier.h"

KNNClassifier::KNNClassifier() {
	k = 1;
}
KNNClassifier::KNNClassifier(vector<MyData> X, int k):BaseClassifier(X) {
	this->k = k;
}

bool compfunc(pair<int, double> a, pair<int, double> b) {
	return a.second < b.second;
}

int KNNClassifier::prediction(MyData t) {

	int vsize = X.size();
	if (vsize == 0) {
		cout << "Prediciton error, not enough data.\n";
		return -1;
	}

	vector<pair<int, double>> dis_matrix;

	for (int i = 0; i < vsize; i++) {
		dis_matrix.push_back(pair<int, double>(X[i].label, euDistance(t, X[i])));
	}

	partial_sort(dis_matrix.begin(),dis_matrix.begin()+k, dis_matrix.end(), compfunc);

	int fcount = 1, maxfcount = 1;
	int pre_class = dis_matrix[0].first;
	int max_class = pre_class;

	for (int i = 1; i < k; i++) {
		if (dis_matrix[i].first == pre_class) {
			fcount++;
			if (fcount > maxfcount) {
				maxfcount = fcount;
				max_class = pre_class;
			}
		}
		else {
			fcount = 1;
			pre_class = dis_matrix[i].first;
		}
	}

	return max_class;
}

vector<int> KNNClassifier::prediction(vector<MyData> T) {

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

void KNNClassifier::setK(int k) {
	this->k = k;
}