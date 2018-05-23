#include<iostream>
#include"KnnBayesSemi.h"
#include"NonlinearSemi.h"
#include"Utility.h"
#include"MyData.h"
#include"ClusterSemi.h"


using namespace std;
int main() {
	string prefix = "C:\\Users\\steven954211\\Desktop\\testData2\\";
	string folder = "C:\\Users\\steven954211\\Documents\\GitHub\\CS_project_ML\\plot\\matrix\\";
	string datalist[] = {
		"ecoli_s" };
	TransD(prefix, folder, datalist);

	

	return 0;
}
