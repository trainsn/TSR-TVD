#include <stdio.h>
#include <sstream> 
#include <string>
#include <stdlib.h>
#include <time.h>
using namespace std;

const int xiVSize = 480;
const int yiVSize = 720;
const int ziVSize = 120;

const int xSubSize = 64;
const int ySubSize = 64;
const int zSubSize = 64;
FILE *fp;
FILE *fp_list;

// load raw data
char root[256] = "../exavisData/combustion/";
const int train_end = 39;
const int test_start = 112;
const int test_end = 112;
const int timestep = 11;
const int train_dataSize = 400;
const int test_xTimes = 30;
const int test_yTimes = 45;
const int test_zTimes = 7;
const int test_dataSize = test_xTimes * test_yTimes * test_zTimes;

int train_xstart[train_dataSize][train_end][timestep];
int train_ystart[train_dataSize][train_end][timestep];
int train_zstart[train_dataSize][train_end][timestep];
int train_xmin[train_dataSize][train_end][timestep];
int train_ymin[train_dataSize][train_end][timestep];
int train_zmin[train_dataSize][train_end][timestep];
float train_Vmin[train_dataSize][train_end][timestep];
int train_xmax[train_dataSize][train_end][timestep];
int train_ymax[train_dataSize][train_end][timestep];
int train_zmax[train_dataSize][train_end][timestep];
float train_Vmax[train_dataSize][train_end][timestep];

int test_xstart[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_ystart[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_zstart[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_xmin[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_ymin[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_zmin[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
float test_Vmin[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_xmax[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_ymax[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
int test_zmax[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];
float test_Vmax[test_dataSize][(test_end - test_start) / (timestep - 1) + 1];

void CropOneIns(int z_start, int y_start, int x_start, int time_start, int t, float** raw_data, bool train, int d) {
	float *sub_data;
	sub_data = new float[xSubSize * ySubSize * zSubSize];

	float maxV = -1.0, minV = 2.0;
	int subIdx = 0;
	int maxz = -1, maxy = -1, maxx = -1, minz = -1, miny = -1, minx = -1;
	for (int i = z_start; i < z_start + zSubSize; i++) {
		for (int j = y_start; j < y_start + ySubSize; j++) {
			for (int k = x_start; k < x_start + xSubSize; k++) {
				int idx = i * yiVSize*xiVSize + j * xiVSize + k;
				subIdx = (i - z_start) * ySubSize*xSubSize + (j - y_start) * xSubSize + (k - x_start);
				sub_data[subIdx] = raw_data[t][idx];
				if (sub_data[subIdx] > maxV) {
					maxV = raw_data[t][idx];
					maxz = i - z_start; maxy = j - y_start; maxx = k - x_start;
				}

				if (sub_data[subIdx] < minV) {
					minV = raw_data[t][idx];
					minz = i - z_start; miny = j - y_start; minx = k - x_start;
				}
			}
		}
	}

	if (train) {
		train_xstart[d][time_start][t] = x_start;
		train_ystart[d][time_start][t] = y_start;
		train_zstart[d][time_start][t] = z_start;
		train_xmin[d][time_start][t] = minx;
		train_ymin[d][time_start][t] = miny;
		train_zmin[d][time_start][t] = minz;
		train_Vmax[d][time_start][t] = minV;
		train_xmax[d][time_start][t] = maxx;
		train_ymax[d][time_start][t] = maxy;
		train_zmax[d][time_start][t] = maxz;
		train_Vmax[d][time_start][t] = maxV;
	}
	else {
		test_xstart[d][(time_start - test_start) / (timestep - 1)] = x_start;
		test_ystart[d][(time_start - test_start) / (timestep - 1)] = y_start;
		test_zstart[d][(time_start - test_start) / (timestep - 1)] = z_start;
		test_xmin[d][(time_start - test_start) / (timestep - 1)] = minx;
		test_ymin[d][(time_start - test_start) / (timestep - 1)] = miny;
		test_zmin[d][(time_start - test_start) / (timestep - 1)] = minz;
		test_Vmin[d][(time_start - test_start) / (timestep - 1)] = minV;
		test_xmax[d][(time_start - test_start) / (timestep - 1)] = maxx;
		test_ymax[d][(time_start - test_start) / (timestep - 1)] = maxy;
		test_zmax[d][(time_start - test_start) / (timestep - 1)] = maxz;
		test_Vmax[d][(time_start - test_start) / (timestep - 1)] = maxV;
	}

	// save in file 
	ostringstream os;
	if (train) {
		if (time_start + t < 10)
			os << root << "train_cropped/" << "jet_mixfrac_000" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
		else if (time_start + t < 100)
			os << root << "train_cropped/" << "jet_mixfrac_00" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
		else
			os << root << "train_cropped/" << "jet_mixfrac_0" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
	}
	else {
		if (time_start + t < 10)
			os << root << "test_cropped/" << "jet_mixfrac_000" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
		else if (time_start + t < 100)
			os << root << "test_cropped/" << "jet_mixfrac_00" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
		else
			os << root << "test_cropped/" << "jet_mixfrac_0" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";

	}

	string outFilename(os.str());
	fp = fopen(outFilename.c_str(), "wb");
	if (fp == NULL) {
		printf("can't open file %s\n", outFilename);
	}
	fwrite(sub_data, sizeof(float), xSubSize * ySubSize * zSubSize, fp);
	delete[]sub_data;
	fclose(fp);
}


void CropData(int time_start, bool train) {
	float *raw_data[timestep];
	for (int t = time_start; t < time_start + timestep && t <= test_end; t++) {
		ostringstream os;
		if (t < 10)
			os << root << "jet_000" << t << "/jet_mixfrac_000" << t << ".dat";
		else if (t < 100)
			os << root << "jet_00" << t << "/jet_mixfrac_00" << t << ".dat";
		else
			os << root << "jet_0" << t << "/jet_mixfrac_0" << t << ".dat";
		string filename(os.str());
		fp = fopen(filename.c_str(), "rb");
		if (fp == NULL) {
			printf("can't open file %s\n", filename);
		}

		raw_data[t - time_start] = new float[xiVSize * yiVSize * ziVSize];
		fread(raw_data[t - time_start], sizeof(float), xiVSize * yiVSize * ziVSize, fp);
		fclose(fp);
	}

	// crop data
	if (train) {
		for (int d = 0; d < train_dataSize; d++) {
			int x_start = rand() % (xiVSize - xSubSize + 1);
			int y_start = rand() % (yiVSize - ySubSize + 1);
			int z_start = rand() % (ziVSize - zSubSize + 1);

			for (int t = 0; t < timestep; t++) {
				CropOneIns(z_start, y_start, x_start, time_start, t, raw_data, train, d);
			}
		}
	}
	else {
		for (int zc = 0; zc < test_zTimes; zc++) {
			int z_start = (int)(float((ziVSize - zSubSize)) / ((test_zTimes - 1) ? (test_zTimes - 1) : 1) * zc);
			for (int yc = 0; yc < test_yTimes; yc++) {
				int y_start = (int)(float((yiVSize - ySubSize)) / ((test_yTimes - 1) ? (test_yTimes - 1) : 1) * yc);
				for (int xc = 0; xc < test_xTimes; xc++) {					
					int x_start = (int)(float((xiVSize - xSubSize)) / ((test_xTimes - 1) ? (test_xTimes - 1) : 1) * xc);
					int d = zc * test_yTimes*test_xTimes + yc * test_xTimes + xc;
					CropOneIns(z_start, y_start, x_start, time_start, 0, raw_data, train, d);
				}
			}
		}
	}	

	for (int t = 0; t < timestep; t++)
		delete[]raw_data[t];
}


int main() {
	srand(time(0));


	// training data prep
	ostringstream os;
	os.str("");
	os << root << "train_cropped/" << "volume_train_list.txt";
	string volume_list(os.str());
	os.str("");
	os << root << "train_cropped/" << "volume_train_statis.txt";
	string volume_statis(os.str());
	freopen(volume_statis.c_str(), "w", stdout);

	fp_list = fopen(volume_list.c_str(), "w");
	fprintf(fp_list, "%d\n", train_dataSize);
	fprintf(fp_list, "%d\n", train_end);

	for (int time_start = 1; time_start <= train_end; time_start++) {
		CropData(time_start, true);
	}
	
	// output filenames and statistic information
	for (int d=0; d<train_dataSize;d++)
		for (int time_start = 1; time_start <= train_end; time_start++)
			for (int t = 0; t < timestep; t++) {
				printf("%d %d %d %d %d %d %d %d %f\n", d, time_start + t, 
					train_xstart[d][time_start][t], train_ystart[d][time_start][t], train_zstart[d][time_start][t], 
					train_xmin[d][time_start][t], train_ymin[d][time_start][t], train_zmin[d][time_start][t], 
					train_Vmin[d][time_start][t]);
				printf("%d %d %d %d %d %d %d %d %f\n", d, time_start + t,
					train_xstart[d][time_start][t], train_ystart[d][time_start][t], train_zstart[d][time_start][t],
					train_xmax[d][time_start][t], train_ymax[d][time_start][t], train_zmax[d][time_start][t],
					train_Vmax[d][time_start][t]);
				
				os.str("");
				if (time_start + t < 10)
					os << root << "train_cropped/" << "jet_mixfrac_000" << (time_start + t) 
					<< "_x" << train_xstart[d][time_start][t] << "_y" << train_ystart[d][time_start][t] 
					<< "_z" << train_zstart[d][time_start][t] << ".raw";
				else if (time_start + t < 100)
					os << root << "train_cropped/" << "jet_mixfrac_00" << (time_start + t) 
					<< "_x" << train_xstart[d][time_start][t] << "_y" << train_ystart[d][time_start][t]
					<< "_z" << train_zstart[d][time_start][t] << ".raw"; 
				else
					os << root << "train_cropped/" << "jet_mixfrac_0" << (time_start + t) 
					<< "_x" << train_xstart[d][time_start][t] << "_y" << train_ystart[d][time_start][t]
					<< "_z" << train_zstart[d][time_start][t] << ".raw";
				string outFilename(os.str());
				fprintf(fp_list, "%s\n", outFilename.substr(outFilename.rfind("/") + 1).c_str());
			}

	// testing data prep 
	//os.str("");
	//os << root << "test_cropped/" << "volume_test_list_" << test_start << "-" << test_end << ".txt";
	//string volume_list(os.str());
	//os.str("");
	//os << root << "test_cropped/" << "volume_test_statis_" << test_start << "-" << test_end << ".txt";
	//string volume_statis(os.str());
	//freopen(volume_statis.c_str(), "w", stdout);

	//fp_list = fopen(volume_list.c_str(), "w");
	//fprintf(fp_list, "%d\n", test_dataSize);
	////fprintf(fp_list, "%d\n", (test_end - test_start) / (timestep - 1) + 1);

	//for (int time_start = test_start; time_start <= test_end; time_start += timestep - 1) {
	//	CropData(time_start, false);
	//}
	//
	//for (int time_start = test_start; time_start < test_end; time_start += timestep - 1)
	//	for (int t = 0; t < timestep; t += timestep - 1)
	//		for (int d = 0; d < test_dataSize; d++)		
	//		 {
	//			printf("%d %d %d %d %d %d %d %d %f\n", d, time_start + t,
	//				test_xstart[d][(time_start - test_start) / (timestep - 1)], test_ystart[d][(time_start - test_start) / (timestep - 1)], test_zstart[d][(time_start - test_start) / (timestep - 1)],
	//				test_xmin[d][(time_start - test_start) / (timestep - 1)], test_ymin[d][(time_start - test_start) / (timestep - 1)], test_zmin[d][(time_start - test_start) / (timestep - 1)],
	//				test_Vmin[d][(time_start - test_start) / (timestep - 1)]);
	//			printf("%d %d %d %d %d %d %d %d %f\n", d, time_start + t,
	//				test_xstart[d][(time_start - test_start) / (timestep - 1)], test_ystart[d][(time_start - test_start) / (timestep - 1)], test_zstart[d][(time_start - test_start) / (timestep - 1)],
	//				test_xmax[d][(time_start - test_start) / (timestep - 1)], test_ymax[d][(time_start - test_start) / (timestep - 1)], test_zmax[d][(time_start - test_start) / (timestep - 1)],
	//				test_Vmax[d][(time_start - test_start) / (timestep - 1)]);

	//			os.str("");
	//			if (time_start + t < 10)
	//				os << root << "test_cropped/" << "jet_mixfrac_000" << (time_start + t)
	//				<< "_x" << test_xstart[d][(time_start - test_start) / (timestep - 1)] << "_y" << test_ystart[d][(time_start - test_start) / (timestep - 1)]
	//				<< "_z" << test_zstart[d][(time_start - test_start) / (timestep - 1)] << ".raw";
	//			else if (time_start + t < 100)
	//				os << root << "test_cropped/" << "jet_mixfrac_00" << (time_start + t)
	//				<< "_x" << test_xstart[d][(time_start - test_start) / (timestep - 1)] << "_y" << test_ystart[d][(time_start - test_start) / (timestep - 1)]
	//				<< "_z" << test_zstart[d][(time_start - test_start) / (timestep - 1)] << ".raw";
	//			else
	//				os << root << "test_cropped/" << "jet_mixfrac_0" << (time_start + t)
	//				<< "_x" << test_xstart[d][(time_start - test_start) / (timestep - 1)] << "_y" << test_ystart[d][(time_start - test_start) / (timestep - 1)]
	//				<< "_z" << test_zstart[d][(time_start - test_start) / (timestep - 1)] << ".raw";
	//			string outFilename(os.str());
	//			fprintf(fp_list, "%s\n", outFilename.substr(outFilename.rfind("/") + 1).c_str());
	//		}
}