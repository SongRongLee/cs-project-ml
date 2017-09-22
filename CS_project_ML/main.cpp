#include<iostream>
#include"KNNClassifier.h"
#include"Utility.h"
#include"MyData.h"
using namespace std;

int main() {

	//---user define params---
	string dirname = "C:\\Users\\Hubert_Lee\\Desktop\\CS_project\\d1-7_s\\d1_s";
	int k = 3;
	//------------------------

	float validation_err = 0;
	float accuracy;
	int wrong_count = 0;

	for (int i = 1; i <= 10; i++) {

		vector<MyData> X;
		vector<MyData> T;
		vector<int> result;

		extractData(X, T, dirname, i);

		//testing
		KNNClassifier knn(X ,k);
		result = knn.prediction(T);

		//printing result
		wrong_count = checkResult(result, T);
		validation_err += wrong_count;
		accuracy = (float)(T.size() - wrong_count) / (float)T.size() * 100;
		cout << "Fold " << i << " done with accuracy " << accuracy << "%" << endl;

	}

	validation_err /= 10;
	cout << "Data analyzing done." << endl;
	cout << "Model validation value = " << validation_err << endl;
	system("pause");

	return 0;
}