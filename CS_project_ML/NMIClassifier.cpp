#include"NMIClassifier.h"

NMIClassifier::NMIClassifier() {
	k = 1;
}
NMIClassifier::NMIClassifier(vector<MyData> X, int k) :BaseClassifier(X) {
	this->k = k;
}

void NMIClassifier::setK(int k) {
	this->k = k;
}

double NMIClassifier::euDistance(MyData a, MyData b) {

	if (a.features.size() != b.features.size()) {
		cout << "Euclidean distance error, size mismatch.\n";
		return -1;
	}

	int vsize = a.features.size();
	double tempsquare = 0;
	for (int i = 0; i < vsize; i++) {
		tempsquare += pow(a.features[i] - b.features[i], 2);
	}
	return sqrt(tempsquare);
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
void NMIClassifier::compute_medoid()
{

	vector<int> cker;
	for (int i = 0; i < this->X.size(); i++)
	{
		int flag = 0;
		for (int j = 0; j < cker.size(); j++)
		{
			if (X[i].label == cker[j])
			{
				flag = 1;
			}
		}
		if (flag == 0)
		{
			cker.push_back(X[i].label);
			this->medoid.push_back(pair<int, double>(X[i].label, -1));
			this->medoid_idx.push_back(pair<int,int>(X[i].label, 0));
		}
	}
	for (int i = 0; i < this->X.size(); i++)
	{
		double dis_sum = 0;
		int label_idx;
		for (int j = 0; j < this->X.size(); j++)
		{
			if (i != j)
			{
				if (X[i].label == X[j].label)
				{
					dis_sum += euDistance(X[i], X[j]);
				}
			}
		}
		for (int j = 0; j < this->medoid.size(); j++)
		{
			if (X[i].label == this->medoid[j].first)
			{
				label_idx = j;
			}
		}
		if (dis_sum < this->medoid[label_idx].second || this->medoid[label_idx].second < 0)
		{
			this->medoid[label_idx].second = dis_sum;
			this->medoid_idx[label_idx].second = i;
		}
	}
}
int NMIClassifier::prediction(MyData t)
{
	double min_dis = -1;
	int min_label = 0;
	/*for (int i = 0; i < medoid.size(); i++)
	{
		cout << medoid[i].first << " " << medoid[i].second << " " << medoid_idx[i].second << endl;
	}*/
	for (int i = 0; i < medoid.size(); i++)
	{
		if (euDistance(X[medoid_idx[i].second],t) < min_dis || min_dis < 0)
		{
			min_dis = medoid[i].second;
			min_label = medoid[i].first;
		}
	}
	return min_label;
}