#include<iostream>
#include"KNNClassifier.h"
#include"Utility.h"
using namespace std;

int main() {

	string dirname = "C:\\Users\\Hubert_Lee\\Desktop\\CS_project\\d1-7_s\\d1_s";
	float validation_err = 0;
	float accuracy;
	int wrong_count = 0;

	for (int i = i; i <= 10; i++) {

		vector<vector<float>> X;
		vector<vector<float>> TX;
		vector<int> Y;
		vector<int> TY;
		vector<int> result;

		extractData(X, Y, TX, TY, dirname);

		KNNClassifier knn(X, Y);
		result = knn.prediction(TX);

		wrong_count = checkResult(result, TY);
		validation_err += wrong_count;
		accuracy = (TX.size() - wrong_count) / TX.size();
		cout << "Fold " << i << " done with accuracy " << accuracy << "%" << endl;
	}

	validation_err /= 10;
	cout << "Data analyzing done." << endl;
	cout << "Model validation value = " << validation_err << endl;
	system("pause");

	return 0;
}