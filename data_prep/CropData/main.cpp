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
char root[256] = "D:\\OSU\\Grade1\\Research\\TSR-TVD\\exavisData\\combustion\\";
const int train_end = 45;
const int test_start = 50;
const int test_end = 118;
const int timestep = 5;
const int dataSize = 100;

int train_xstart[dataSize][train_end][timestep];
int train_ystart[dataSize][train_end][timestep];
int train_zstart[dataSize][train_end][timestep];
int train_xmin[dataSize][train_end][timestep];
int train_ymin[dataSize][train_end][timestep];
int train_zmin[dataSize][train_end][timestep];
float train_Vmin[dataSize][train_end][timestep];
int train_xmax[dataSize][train_end][timestep];
int train_ymax[dataSize][train_end][timestep];
int train_zmax[dataSize][train_end][timestep];
float train_Vmax[dataSize][train_end][timestep];

void CropData(int time_start) {
	float *raw_data[timestep];
	for (int t = time_start; t < time_start + timestep; t++) {
		ostringstream os;
		if (t < 10)
			os << root << "jet_000" << t << "\\jet_mixfrac_000" << t << ".dat";
		else if (t < 100)
			os << root << "jet_00" << t << "\\jet_mixfrac_00" << t << ".dat";
		else
			os << root << "jet_0" << t << "\\jet_mixfrac_0" << t << ".dat";
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
	for (int d = 0; d < dataSize; d++) {
		int x_start = rand() % (xiVSize - xSubSize + 1);
		int y_start = rand() % (yiVSize - ySubSize + 1);
		int z_start = rand() % (ziVSize - zSubSize + 1);

		for (int t = 0; t < timestep; t++) {
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
			
			// save in file 
			ostringstream os;
			if (time_start + t < 10)
				os << root << "cropped\\" << "jet_mixfrac_000" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
			else if (time_start + t < 100)
				os << root << "cropped\\" << "jet_mixfrac_00" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
			else 
				os << root << "cropped\\" << "jet_mixfrac_0" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";

			string outFilename(os.str());
			fp = fopen(outFilename.c_str(), "wb");
			if (fp == NULL) {
				printf("can't open file %s\n", outFilename);
			}
			fwrite(sub_data, sizeof(float), xSubSize * ySubSize * zSubSize, fp);
			delete[]sub_data;
			fclose(fp);
		}
	}
	for (int t = 0; t < timestep; t++)
		delete[]raw_data[t];
}


int main() {
	srand(time(0));

	ostringstream os;
	os.str("");
	os << root << "cropped\\" << "volume_list.txt";
	string volume_list(os.str());
	os.str("");
	os << root << "cropped\\" << "volume_statis.txt";
	string volume_statis(os.str());
	freopen(volume_statis.c_str(), "w", stdout);

	fp_list = fopen(volume_list.c_str(), "w");
	fprintf(fp_list, "%d\n", dataSize);

	for (int time_start = 1; time_start <= train_end; time_start++) {
		CropData(time_start);
	}
	
	// output filenames and statistic information
	for (int d=0; d<dataSize;d++)
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
					os << root << "cropped\\" << "jet_mixfrac_000" << (time_start + t) 
					<< "_x" << train_xstart[d][time_start][t] << "_y" << train_ystart[d][time_start][t] 
					<< "_z" << train_zstart[d][time_start][t] << ".raw";
				else if (time_start + t < 100)
					os << root << "cropped\\" << "jet_mixfrac_00" << (time_start + t) 
					<< "_x" << train_xstart[d][time_start][t] << "_y" << train_ystart[d][time_start][t]
					<< "_z" << train_zstart[d][time_start][t] << ".raw"; 
				else
					os << root << "cropped\\" << "jet_mixfrac_0" << (time_start + t) 
					<< "_x" << train_xstart[d][time_start][t] << "_y" << train_ystart[d][time_start][t]
					<< "_z" << train_zstart[d][time_start][t] << ".raw";
				string outFilename(os.str());
				fprintf(fp_list, "%s\n", outFilename.substr(outFilename.rfind("\\") + 1).c_str());
			}

}