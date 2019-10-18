#include <stdio.h>
#include <sstream> 
#include <string>
#include <stdlib.h>
#include <time.h>
using namespace std;

int main() {
	char filename[256];
	scanf("%s", filename);

	const int xiVSize = 480;
	const int yiVSize = 720;
	const int ziVSize = 120;

	const int xSubSize = 64;
	const int ySubSize = 64;
	const int zSubSize = 64;

	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		printf("can't open file %s\n", filename);
	}

	float *data; 
	data = new float[xiVSize * yiVSize * ziVSize];
	fread(data, sizeof(float), xiVSize * yiVSize * ziVSize, fp);
	float *subData;
	subData = new float[xSubSize * ySubSize * zSubSize];

	srand(time(0));
	int x_start = rand() % (xiVSize - xSubSize + 1);
	int y_start = rand() % (yiVSize - ySubSize + 1);
	int z_start = rand() % (ziVSize - zSubSize + 1);

	float maxV = -1.0, minV = 2.0;
	int subIdx = 0;
	int maxz = -1, maxy = -1, maxx = -1, minz = -1, miny = -1, minx = -1;
	for (int i = 0; i < ziVSize; i++) {
		for (int j = 0; j < yiVSize; j++) {
			for (int k = 0; k < xiVSize; k++) {
				int idx = i * yiVSize*xiVSize + j * xiVSize + k;
				if (i >= z_start && i < z_start + zSubSize && j >= y_start && j < y_start + ySubSize && k >= x_start && k < x_start + xSubSize) {
					subIdx = (i-z_start) * ySubSize*xSubSize + (j-y_start) * xSubSize + (k-x_start);
					subData[subIdx] = data[idx];
					if (subData[subIdx] > maxV) {
						maxV = data[idx];
						maxz = i - z_start; maxy = j - y_start; maxx = k - x_start;
					}
						
					if (subData[subIdx] < minV) {
						minV = data[idx];
						minz = i - z_start; miny = j - y_start; minx = k - x_start;
					}
				}			
			}
		}
	}
	fclose(fp);
	printf("%d %d %d %f\n", minx, miny, minz, minV);
	printf("%d %d %d %f\n", maxx, maxy, maxz, maxV);

	// save in file 
	string filePath(filename);
	filePath = filePath.substr(0, filePath.rfind("."));
	ostringstream os; 
	os << filePath << "_x" <<  x_start << "_y" << y_start << "_z" << z_start << ".raw";
	string outFilename(os.str());

	fp = fopen(outFilename.c_str(), "wb");
	if (fp == NULL) {
		printf("can't open file %s\n", outFilename);
	}
	fwrite(subData, sizeof(float), xSubSize * ySubSize * zSubSize, fp);
	system("pause");
}