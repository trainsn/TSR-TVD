#include <stdio.h>
#include <sstream> 
#include <string>
#include <stdlib.h>
#include <time.h>
using namespace std;

int main() {
	srand(time(0));

	const int xiVSize = 480;
	const int yiVSize = 720;
	const int ziVSize = 120;

	const int xSubSize = 480;
	const int ySubSize = 720;
	const int zSubSize = 120;
	FILE *fp;
	FILE *fp_list;

	// load raw data
	char root[256] = "D:\\OSU\\Grade1\\Research\\TSR-TVD\\exavisData\\combustion\\Jet_0016-0020\\";
	ostringstream os;
	const int timestep = 5;
	const int time_start = 16;
	float *data[timestep];
	for (int t = time_start; t < time_start + timestep; t++) {
		os.str("");
		os << root << "jet_00" << t << "\\jet_mixfrac_00" << t << ".raw";
		string filename(os.str());
		fp = fopen(filename.c_str(), "rb");
		if (fp == NULL) {
			printf("can't open file %s\n", filename);
		}

		data[t - time_start] = new float[xiVSize * yiVSize * ziVSize];
		fread(data[t - time_start], sizeof(float), xiVSize * yiVSize * ziVSize, fp);
		fclose(fp);
	}

	// crop data
	const int dataSize = 1;
	const int max_step = 3;
	os.str("");
	os << root << "cropped\\" << "volume_list.txt";
	string volume_list(os.str());
	os.str("");
	os << root << "cropped\\" << "volume_statis.txt";
	string volume_statis(os.str());
	freopen(volume_statis.c_str(), "w", stdout);

	fp_list = fopen(volume_list.c_str(), "w");
	fprintf(fp_list, "%d\n", dataSize);
	
	for (int d = 0; d < dataSize; d++) {
		int x_start = rand() % (xiVSize - xSubSize + 1);
		int y_start = rand() % (yiVSize - ySubSize + 1);
		int z_start = rand() % (ziVSize - zSubSize + 1);

		fprintf(fp_list, "%d\n", timestep);
		for (int t = 0; t < timestep; t++) {
			float *subData;
			subData = new float[xSubSize * ySubSize * zSubSize];

			float maxV = -1.0, minV = 2.0;
			int subIdx = 0;
			int maxz = -1, maxy = -1, maxx = -1, minz = -1, miny = -1, minx = -1;
			for (int i = 0; i < ziVSize; i++) {
				for (int j = 0; j < yiVSize; j++) {
					for (int k = 0; k < xiVSize; k++) {
						int idx = i * yiVSize*xiVSize + j * xiVSize + k;
						if (i >= z_start && i < z_start + zSubSize && j >= y_start && j < y_start + ySubSize && k >= x_start && k < x_start + xSubSize) {
							subIdx = (i - z_start) * ySubSize*xSubSize + (j - y_start) * xSubSize + (k - x_start);
							subData[subIdx] = data[t][idx];
							if (subData[subIdx] > maxV) {
								maxV = data[t][idx];
								maxz = i - z_start; maxy = j - y_start; maxx = k - x_start;
							}

							if (subData[subIdx] < minV) {
								minV = data[t][idx];
								minz = i - z_start; miny = j - y_start; minx = k - x_start;
							}
						}
					}
				}
			}
			printf("%d %d %d %d %d %d %d %d %f\n", d, time_start + t, x_start, y_start, z_start, minx, miny, minz, minV);
			printf("%d %d %d %d %d %d %d %d %f\n", d, time_start + t, x_start, y_start, z_start, maxx, maxy, maxz, maxV);

			// save in file 
			//os.str("");
			//os << root << "cropped\\" << "jet_mixfrac_00" << (time_start + t) << "_x" << x_start << "_y" << y_start << "_z" << z_start << ".raw";
			//string outFilename(os.str());
			//fprintf(fp_list, "%s\n", outFilename.substr(outFilename.rfind("\\") + 1).c_str());

			//fp = fopen(outFilename.c_str(), "wb");
			//if (fp == NULL) {
			//	printf("can't open file %s\n", outFilename);
			//}
			//fwrite(subData, sizeof(float), xSubSize * ySubSize * zSubSize, fp);
			//delete[]subData;
			//fclose(fp);
		}
	}
	system("pause");
}