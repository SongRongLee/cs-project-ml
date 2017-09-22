#ifndef UTILITY_H
#define UTILITY_H

#include<iostream>
#include<fstream>
#include<vector>
#include<string>

using namespace std;

void extractData(vector<vector<float>> &X, vector<int> &Y, vector<vector<float>> T, vector<int> TY, string dirname);
int checkResult(vector<int> result, vector<int> correct);

#endif
