#ifndef MY_DATA_H
#define MY_DATA_H
#include<iostream>
#include<vector>
using namespace std;
class MyData
{
friend ostream& operator << (ostream& out, MyData& d);
public:
	vector<double> features;
	int num, label, knn_label, real_label;
	double class_w;
	bool is_train;
	vector<pair<int, double>> class_w_table;
public:
	MyData();
	MyData(int num, vector<double> features, int label, int real_label, bool is_train);
};

#endif 
