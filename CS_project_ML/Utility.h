#ifndef UTILITY_H
#define UTILITY_H

#include<iostream>
#include<fstream>
#include<iomanip>
#include<vector>
#include<string>
#include"MyData.h"

#define EU_DIS 0

using namespace std;

void extractData(vector<MyData> &X, vector<MyData> &T, string dirname, int foldnum);
int checkResult(vector<int> &result, vector<MyData> &T);
double euDistance(MyData a, MyData b);
void genDismatrix(vector<MyData> &X, vector<vector<double>> &dis_matrix, int dis_type = EU_DIS);
void printDismatrix(vector<vector<double>> &dis_matrix);

#endif
