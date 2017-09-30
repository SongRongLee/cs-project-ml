#include"BaseClassifier.h"

BaseClassifier::BaseClassifier() {
	data_count = 0;
}

BaseClassifier::BaseClassifier(vector<MyData> &X) {
	data_count = 0;
	this->X = X;
}

void BaseClassifier::addData(MyData x) {
	X.push_back(x);
}