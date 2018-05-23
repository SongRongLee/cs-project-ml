#ifndef UTILITY_H
#define UTILITY_H

#include<iostream>
#include<fstream>
#include<iomanip>
#include<vector>
#include<string>
#include"MyData.h"
#include <Eigen/Dense>
#define EU_DIS 0

using namespace std;

void extractData(vector<MyData> &X, vector<MyData> &T, string dirname, int foldnum);
//for semi-supervised
void extractData(vector<MyData> &X, vector<MyData> &XT, vector<MyData> &T, string dataname, string labelname);


//comp functions
bool mycomp(pair<int, double> a, pair<int, double> b);
bool mycomp_label(pair<int, double> a, pair<int, double> b);
bool mycomp_index(MyData a, MyData b);
bool compfunc_mydata(pair<MyData, double> a, pair<MyData, double> b);
bool compfunc_dispair(pair<vector<pair<int, double>>, double> a, pair<vector<pair<int, double>>, double> b);
bool compfunc_descend(pair<int, double> a, pair<int, double> b);

int checkResult(vector<int> &result, vector<MyData> &T);
double calDistance(MyData a, MyData b, int dis_type);
double euDistance(MyData a, MyData b);
void genDismatrix(vector<MyData> &X, vector<vector<double>> &dis_matrix, int dis_type = EU_DIS);
void indexSortedMatrix(vector<MyData> &total_data, vector<vector<double>> &dis_matrix, vector<vector<double>> &new_dis);
void indexSortedAllMatrix(vector<MyData> &total_data, vector<vector<vector<double>>> &dis_matrixs, vector<vector<vector<double>>>&new_diss);
void printDismatrix(vector<vector<double>> &dis_matrix);
void CreateFolder(const string  path);
void printDismatrix(vector<vector<double>> &dis_matrix, ofstream &out);
void printTestDis(vector<vector<vector<double>>> dis_matrixs,int num, const vector<MyData> &total_data);
void printTestDis(vector<vector<vector<double>>> dis_matrixs, int num, const vector<MyData> &total_data, ofstream &out);
void printlabel(vector<MyData> &total_data, ofstream &outknn, ofstream &outreal);
void printlabel(vector<MyData>& total_data, ofstream &outknn, ofstream &outreal, vector<int> results);
#endif
