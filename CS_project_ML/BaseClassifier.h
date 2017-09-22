#ifndef BASE_CLASSIFIER_H
#define BASE_CLASSIFIER_H

#include<vector>
#include<string>
using namespace std;

class BaseClassifier
{
protected:
	int data_count;
	vector<vector<float>> X;
	vector<int> Y;
public:
	BaseClassifier();
	BaseClassifier(vector<vector<float>> X, vector<int> Y);
	void addData(vector<float> x, int y);
	virtual int prediction(vector<float> t) = 0;
	virtual vector<int> prediction(vector<vector<float>> TX) = 0;
};

#endif