#include"MyData.h"

MyData::MyData() {}

MyData::MyData(int num, vector<double> features, int label) {
	this->num = num;
	this->features = features;
	this->label = label;
}