#include"Utility.h"

void extractData(vector<vector<float>> &X, vector<int> &Y, vector<vector<float>> T, vector<int> TY, string dirname) {
//todo
}

int checkResult(vector<int> result, vector<int> correct) {
	int vsize = result.size();
	int ans = 0;
	for (int i = 0; i < vsize; i++) {
		if (result[i] != correct[i]) ans++;
	}
	return ans;
}