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
	int num, label;
public:
	MyData();
	MyData(int num, vector<double> features, int label);	
};

#endif 
