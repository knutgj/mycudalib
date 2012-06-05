#ifndef MYCUDALIB_H
#define MYCUDALIB_H

#include <cuComplex.h>

extern "C" void elementMult(int N, const float* A, const float* B, float* C);
extern "C" void elementMultd(int N, const float* A, float* B);
extern "C" void elementMultc(int N, const cuFloatComplex* A, const cuFloatComplex* B, cuFloatComplex* C);
extern "C" void elementMultcd(int N, const cuFloatComplex* A, cuFloatComplex* B);
extern "C" void cpycomplex(int N, const float* A, cuFloatComplex* B);
extern "C" void cpyreal(int N, const cuFloatComplex* A, float* B);
extern "C" void setizero(int N, float* A, int i);
extern "C" void settoone(int N, float* A);

extern "C" int hello();

#endif /* MYCUDALIB_H */
