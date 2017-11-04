#include"KNNClassifier.h"

KNNClassifier::KNNClassifier() {
	k = 1;
}
KNNClassifier::KNNClassifier(vector<MyData> &X, int k):BaseClassifier(X) {
	this->k = k;
}

bool compfunc(pair<int, double> a, pair<int, double> b) {
	return a.second < b.second;
}

bool compfunc2(pair<vector<pair<int, double>>, double> a, pair<vector<pair<int, double>>, double> b)
{
	return a.second < b.second;
}

void KNNClassifier::bayesprediction(MyData &t, vector<double> dis_vector)
{
	int vsize = X.size();
	if (vsize == 0) {
		cout << "Prediciton error, not enough data.\n";
		return;
	}
	vector<pair<vector<pair<int,double>>, double>> dis_pair;
	for (int i = 0; i < vsize; i++) {
		dis_pair.push_back(pair<vector<pair<int, double>>, double>(X[i].class_w_table, dis_vector[i]));
	}
	vector<pair<int, double> > labeled_ratio;
	for(int i=0;i<vsize;i++)
	{
		int f = -1;
		for (int j = 0; j < labeled_ratio.size(); j++)
		{
			if (X[i].label == labeled_ratio[j].first)
			{
				f = j;
			}
		}
		if (f == -1)
		{
			labeled_ratio.push_back(pair<int, double>(X[i].label, 1));
		}
		else
		{
			labeled_ratio[f].second++;
		}
	}
	for (int i = 0; i < labeled_ratio.size(); i++)
	{
		labeled_ratio[i].second /= vsize;
	}
	vector<vector<pair<int, double>>> table;
	vector<double> para;
	sort(dis_pair.begin(), dis_pair.end(), compfunc2);
	for (int i = 1; i < vsize; i++)
	{
		double divide = 0;
		vector<pair<int, double>> tmp;
		for (int j = 0; j < i; j++)
		{
			divide += 1 / dis_pair[j].second;
		}
		for (int j = 0; j < i; j++)
		{
			double weight = dis_pair[j].second / divide;
			for (int q = 0; q < dis_pair[j].first.size(); q++)
			{
				int idx = -1;
				for (int p = 0; p <tmp.size(); p++)
				{
					if (dis_pair[j].first[q].first == tmp[p].first)
					{
						idx = p;
					}
				}
				if (idx == -1)
				{
					tmp.push_back(pair<int, double>(dis_pair[j].first[q].first, dis_pair[j].first[q].second*weight));
				}
				else
				{
					tmp[idx].second += dis_pair[j].first[q].second*weight;
				}
			}
		}
		table.push_back(tmp);
	}

	for (int i = 0; i < table.size(); i++)
	{
		double normal = 0;
		for (int j = 0; j < table[i].size(); j++)
		{
			normal += table[i][j].second;
		}
		for (int j = 0; j < table[i].size(); j++)
		{
			table[i][j].second /= normal;
		}
	}
	for (int i = 0; i < table.size(); i++)
	{
		sort(table[i].begin(), table[i].end(), compfunc);
		int id = -1;
		for (int j = 0; j < labeled_ratio.size(); j++)
		{
			if (table[i][0].first = labeled_ratio[j].first)
			{
				id = j;
			}
		}
		if (id != -1)
		{
			para.push_back(labeled_ratio[id].second);
		}
		else
		{
			para.push_back(0);
		}
	}
	vector<pair<int, double>> res;
	for (int i = 0; i < table.size(); i++)
	{
		for (int j = 0; j < table[i].size(); j++)
		{
			int idx = -1;
			for (int q = 0; q < res.size(); q++)
			{
				if (table[i][j].first == res[q].first)
				{
					idx = q;
				}
			}
			if (idx == -1)
			{
				res.push_back(pair<int, double>(table[i][j].first, table[i][j].second*para[i]));
			}
			else
			{
				res[idx].second += table[i][j].second*para[i];
			}
		}
	}
	int max_idx;
	double max = -1;
	for (int i = 0; i < res.size(); i++)
	{
		if (res[i].second > max)
		{
			max = res[i].second;
			max_idx = i;
		}
	}
	t.class_w_table = res;
	t.class_w = max;
	t.knn_label = res[max_idx].first;
}
int KNNClassifier::prediction(MyData &t) {

	int vsize = X.size();
	if (vsize == 0) {
		cout << "Prediciton error, not enough data.\n";
		return -1;
	}

	vector<pair<int, double>> dis_vector;

	for (int i = 0; i < vsize; i++) {
		dis_vector.push_back(pair<int, double>(X[i].label, calDistance(t, X[i], dis_type)));
	}

	partial_sort(dis_vector.begin(), dis_vector.begin()+k, dis_vector.end(), compfunc);

	int fcount = 1, maxfcount = 1;
	int pre_class = dis_vector[0].first;
	int max_class = pre_class;

	for (int i = 1; i < k; i++) {
		if (dis_vector[i].first == pre_class) {
			fcount++;
			if (fcount > maxfcount) {
				maxfcount = fcount;
				max_class = pre_class;
			}
		}
		else {
			fcount = 1;
			pre_class = dis_vector[i].first;
		}
	}

	//set class_weight
	t.class_w = maxfcount / k;

	return max_class;
}

int KNNClassifier::prediction(MyData &t, vector<double> dis_vector) {

	int vsize = X.size();
	if (vsize == 0) {
		cout << "Prediciton error, not enough data.\n";
		return -1;
	}

	vector<pair<int, double>> dis_pair;

	for (int i = 0; i < vsize; i++) {
		dis_pair.push_back(pair<int, double>(X[i].label, dis_vector[i]));
	}

	partial_sort(dis_pair.begin(), dis_pair.begin() + k, dis_pair.end(), compfunc);

	int fcount = 1, maxfcount = 1;
	int pre_class = dis_pair[0].first;
	int max_class = pre_class;

	for (int i = 1; i < k; i++) {
		if (dis_pair[i].first == pre_class) {
			fcount++;
			if (fcount > maxfcount) {
				maxfcount = fcount;
				max_class = pre_class;
			}
		}
		else {
			fcount = 1;
			pre_class = dis_pair[i].first;
		}
	}
	//set class weight
	t.class_w = (double)maxfcount / (double)k;

	return max_class;
}

vector<int> KNNClassifier::prediction(vector<MyData> &T) {

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
