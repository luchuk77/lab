/*Reads the input data for the spheres from file 'energy.inp'  
 * First line contains integer - N, the number of spheres.  
 * Followed by N lines, each contains 4 real numbers: coordinates(x,y,z) and the radius(r). */
                                                                                                                                                     
#include <iostream>
#include <fstream>
#include <random>
#include <omp.h>
#include <cstdlib>
#define M_PI 3.14159265358979323846  
using std::ifstream;
using std::cout;
using std::cerr;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;
const char* name = "energy.inp";
float *x, *y, *z, *r;
unsigned n;
float xL, yL, zL, xR, yR, zR; // The "borders". The farthes away points on spheres of all spheres. 
float xDist, yDist, zDist;
double calculateVolume(unsigned totalSamples);
void readDataForShperes(); void borders(); void borders(float * a, float& l, float& r);
void freeMemory();
int allocateMemory();
void sqrRadius();
double calculateVolume(unsigned totalSamples)
{                                                                                                                                                    
readDataForShperes();                                                                                                                                
if (n <= 0 || totalSamples <= 0) // By default n is 0, so if the file reading fails, it will return 0, which is not so bad...           
return 0.0;
if (n == 1)
return (4.0 / 3.0) * M_PI * r[0] * r[0] * r[0];
borders(); // TODO - with SIMD & threads for the borders...     
xDist = xR - xL;                                                                                                                                     
yDist = yR - yL;                                                                                                                                     
zDist = zR - zL;                                                                                                                                     
// I need to calculate some random Point(a, b, c) distance to (x, y, z, r)      
// So the equation is: sqrt( (a-x)^2 + (b-y)^2 + (c-z)^2 ) <= r         
// Which is equivalent to: (a-x)^2 + (b-y)^2 + (c-z)^2 <= r^2   
// So, I will make the radius it's square at the beginning.     sqrRadius(); 
// TODO - with SIMD & threads...        
// The Monte Carlo      
unsigned i, k, inSphereSamplesFromAllThreads = 0;
int threadsNumber = omp_get_max_threads();
unsigned  * inSphereSamples = (unsigned*)malloc(threadsNumber * sizeof(unsigned));
for (i = 0; i < threadsNumber; ++i)
inSphereSamples[i] = 0;
#pragma omp parallel 
{                                                                                                                                                    
unsigned chunk = totalSamples / threadsNumber;
unsigned end;
unsigned idx = omp_get_thread_num();
unsigned i = idx * chunk, k;
end = (i + chunk) < totalSamples ? i + chunk : totalSamples;                                                                                         
std::default_random_engine generator(idx);                                                                                                           
std::uniform_real_distribution<float> distributionX(xL, xR);
std::uniform_real_distribution<float> distributionY(yL, yR);
std::uniform_real_distribution<float> distributionZ(zL, zR);
float a, b, c, xTmp, yTmp, zTmp, float_rand_max = (float)RAND_MAX;
for (; i < end; ++i)
{                   
a = distributionX(generator);                                                                                                                        
b = distributionY(generator);                                                                                                                        
c = distributionZ(generator);                                                                                                                        
for (k = 0; k < n; ++k)
{
xTmp = x[k] - a;
xTmp *= xTmp;                                                                                                                                        
yTmp = y[k] - b;                                                                                                                                     
yTmp *= yTmp;                                                                                                                                        
zTmp = z[k] - c;                                                                                                                                     
zTmp *= zTmp;                                                                                                                                        
if(xTmp + yTmp + zTmp <= r[k])
{
++inSphereSamples[idx];                                                                                                                              
break;
}
}
}
#pragma omp barrier     
#pragma omp master              
for (i = 0; i < threadsNumber; ++i)
inSphereSamplesFromAllThreads += inSphereSamples[i];                                                                                                 
}                                                                                                                                                    
free(inSphereSamples);                                                                                                                               
return (double)xDist * (double)yDist * (double)zDist * (double)inSphereSamplesFromAllThreads / (double)totalSamples;
}                                                                                                                                                    
// Finds the left and right border on each coordinate 
void borders()
{                                                                                                                                                    
xL = xR = x[0] - r[0];  yL = yR = y[0] - r[0];  zL = zR = z[0] - r[0];
borders(x, xL, xR);         
borders(y, yL, yR);                                                                                                                                  
borders(z, zL, zR); }                                                                                                                                
// Sets the given borders(minimal and maximal value) 
void borders(float * a, float& left, float& right)
{
float tmp;
unsigned i;
for (i = 1; i < n; ++i)
{
tmp = a[i] - r[i];                                                                                                                                   
if (left > tmp)
left = tmp;                                                                                                                                          
tmp = a[i] + r[i];                                                                                                                                   
if (right < tmp)
right = tmp;                                                                                                                                         
}
}
// Makes the values in each field it's square. 
void sqrRadius()
{
unsigned i;
for (i = 0; i < n; ++i)
{
r[i] *= r[i];                                                                                                                                        
}
}
void readDataForShperes()
{                                                                                                                                                    
ifstream in(name);                                                                                                                                   
// First line contains integer - N, the number of spheres.      
in >> n;   
freeMemory();
allocateMemory();                                                                                                                                    
// Followed by N lines, each contains 4 real numbers : coordinates(x, y, z) and the radius(r).  
unsigned i;
for (i = 0; i < n; ++i)
in >> x[i] >> y[i] >> z[i] >> r[i];                                                                                                                  
}
void freeMemory()
{
free(x);                                                                                                                                             
free(y);                                                                                                                                             
free(z);                                                                                                                                             
free(r);                                                                                                                                             
}
int allocateMemory()
{
unsigned int sizeBytes = sizeof(float) * n;
x = (float*)malloc(sizeBytes);
y = (float*)malloc(sizeBytes);
z = (float*)malloc(sizeBytes);
r = (float*)malloc(sizeBytes);
return (x && y && z && r);
}
int main(int argc, char **argv)
{                                                                                                                                                    
// Set the number of threads to use.    
omp_set_dynamic(0);
omp_set_num_threads(atoi(argv[1]));
unsigned int i, samples;
for (samples = 10000000, i = 7; i < 10; samples *= 10, ++i)
{          
double begin = omp_get_wtime();
double res = calculateVolume(samples);
double end = omp_get_wtime();
double elapsed_secs = double(end - begin);
printf("Took %f seconds with 10^%d samples answer is: %.7f\n", elapsed_secs, i, res);
}
return 0;
}