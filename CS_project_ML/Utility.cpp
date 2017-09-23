#include"Utility.h"

string getPrefix(string dirname) {
	int found;
	found = dirname.find_last_of("\\");
	return dirname.substr(found + 1);
}

void extractData(vector<MyData> &X, vector<MyData> &T, string dirname, int foldnum) {
	//configure file names
	string prefix = getPrefix(dirname);
	string datadir = dirname + "\\" + prefix + ".data";
	string folddir = dirname + "\\" + prefix + "_fold" + to_string(foldnum) + ".cv";

	//start reading from .data
	ifstream in(datadir);
	ifstream infold(folddir);
	cout << "reading from " << folddir << endl;

	int data_num, feature_num;
	int fold_data;
	char comma;

	in >> data_num >> comma >> feature_num;
	for (int i = 0; i < data_num; i++) {
		MyData temp_data;
		in >> temp_data.num >> comma;
		infold >> fold_data;
		for (int j = 0; j < feature_num; j++) {
			double temp;
			in >> temp >> comma;
			temp_data.features.push_back(temp);
		}
		in >> temp_data.label;
		if (fold_data == -1) {
			T.push_back(temp_data);
		}
		else{
			X.push_back(temp_data);
		}		
	}
}

int checkResult(vector<int> result, vector<MyData> T) {
	int vsize = result.size();
	int ans = 0;
	for (int i = 0; i < vsize; i++) {
		if (result[i] != T[i].label) ans++;
	}
	return ans;
}

double euDistance(MyData a, MyData b) {

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