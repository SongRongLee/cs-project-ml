#include"MyData.h"

MyData::MyData() {
}

MyData::MyData(int num, vector<double> features, int label, int real_label, bool is_train) {
	this->num = num;
	this->real_label = real_label;
	this->features = features;
	this->label = label;
	this->is_train = is_train;
}

ostream& operator << (ostream &out, MyData &d) {
	out << "Data No:" << d.num << " label:" << d.label << endl;
	out << "features:(";
	for (int i = 0; i < d.features.size()-1; i++) {
		out << d.features[i] << ", ";
	}
	out << d.features.back() << ")" << endl;
	return out;
}