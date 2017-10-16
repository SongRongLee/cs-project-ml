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
//for semi-supervised
void extractData(vector<MyData> &X, vector<MyData> &XT, vector<MyData> &T, string dataname, string labelname);

bool mycomp(pair<int, double> a, pair<int, double> b);
int checkResult(vector<int> &result, vector<MyData> &T);
double calDistance(MyData a, MyData b, int dis_type);
double euDistance(MyData a, MyData b);
void genDismatrix(vector<MyData> &X, vector<vector<double>> &dis_matrix, int dis_type = EU_DIS);
void indexSortedMatrix(vector<MyData> &total_data, vector<vector<double>> &dis_matrix, vector<vector<double>> &new_dis);
void printDismatrix(vector<vector<double>> &dis_matrix);

#endif
