#ifndef UTILITY_H
#define UTILITY_H

#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include"MyData.h"

using namespace std;

void extractData(vector<MyData> &X, vector<MyData> &T, string dirname, int foldnum);
int checkResult(vector<int> result, vector<MyData> T);

#endif
