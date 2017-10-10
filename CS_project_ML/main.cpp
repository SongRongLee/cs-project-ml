#include<iostream>
#include"TransD.h"
#include"Utility.h"
#include"MyData.h"
using namespace std;
int main() {

	//---user define params---
	ofstream out("out.txt");
	string dirname = "C:\\Users\\Hubert_Lee\\Desktop\\CS_project\\d1-7_s\\d1_s";
	int k = 1;
	int fold_num = 1;
	//------------------------

	double validation_err = 0;
	double accuracy;
	int wrong_count = 0;

	for (int i = 1; i <= fold_num; i++) {

		vector<MyData> X;
		vector<MyData> T;
		vector<int> result;
		vector<vector<double>> new_dis;

		extractData(X, T, dirname, i);

		//TransD testing		
		TransD transd(X, T, k);
		transd.performTrans(new_dis);
		for (int j = 0; j < new_dis.size(); j++) {
			for (int k = 0; k < new_dis.size(); k++) {
				int indexj, indexk;
				for (int a = 0; a < X.size(); a++) {
					if (X[a].num == j) {
						indexj = a;
						break;
					}
				}
				for (int a = 0; a < T.size(); a++) {
					if (T[a].num == j) {
						indexj = a + X.size();
						break;
					}
				}
				for (int a = 0; a < X.size(); a++) {
					if (X[a].num == k) {
						indexk = a;
						break;
					}
				}
				for (int a = 0; a < T.size(); a++) {
					if (T[a].num == k) {
						indexk = a + X.size();
						break;
					}
				}
				out << left << fixed << setprecision(6) << setw(9) << new_dis[indexj][indexk];
			}
			out << endl;
		}
		//printDismatrix(new_dis);

		//NMI testing
		/*NMIClassifier nmi(X, 1);
		result = nmi.prediction(T);*/

		//printing result
		/*wrong_count = checkResult(result, T);
		validation_err += wrong_count;
		accuracy = (double)(T.size() - wrong_count) / (double)T.size() * 100;
		cout << "Fold " << i << " done with accuracy " << accuracy << "%" << endl << endl;*/

		//addtional testing
		/*if (i == 1) {
			int vsize = result.size();
			for (int j = 0; j < vsize; j++) {
				if (result[j] != T[j].label) {
					cout << "Data No." << T[j].num << " should be " << T[j].label << endl;
					cout << "But labeled as " << result[j] << endl;
				}
			}
		}*/

		X.clear();
		T.clear();
		wrong_count = 0;
	}

	//validation_err /= 10;
	cout << "Data analyzing done." << endl;
	//cout << "Model validation value = " << validation_err << endl;
	system("pause");

	return 0;
}