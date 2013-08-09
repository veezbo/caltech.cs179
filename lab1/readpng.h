#ifndef READPNG_H
#define READPNG_H

#include <png.h>
using namespace std;

png_bytepp readpng(const char *filename, int* width, int* height);

#endif // READPNG_H
