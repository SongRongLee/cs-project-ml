#include"BaseClassifier.h"
#include"Utility.h"

BaseClassifier::BaseClassifier() {
	dis_type = EU_DIS;
}

BaseClassifier::BaseClassifier(vector<MyData> &X) {
	dis_type = EU_DIS;
	this->X = X;
}

void BaseClassifier::addData(MyData x) {
	X.push_back(x);
}

void BaseClassifier::setDistype(int dis_type) {
	this->dis_type = dis_type;
}