#include"BaseClassifier.h"

BaseClassifier::BaseClassifier() {
	data_count = 0;
}

BaseClassifier::BaseClassifier(vector<vector<float>> X, vector<int> Y) {
	data_count = 0;
	this->X = X;
	this->Y = Y;
}

void BaseClassifier::addData(vector<float> x, int y) {
	X.push_back(x);
	Y.push_back(y);
}