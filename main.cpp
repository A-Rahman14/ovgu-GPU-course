#include <a1.hpp>
#include <iostream>
#include <thread>



using namespace a1;

typedef std::chrono::milliseconds TimeT;

int main(int, char**){
  /*
    Here you should test your code.
    You can access all the functions defined in inc/a1.hpp when specifing the right namespace - here a1.

    for example:
    a1::MatrixMultiplicationSerial(A, B, C, size);
  */



  const unsigned int num_threads = std::thread::hardware_concurrency()/2;
  std::cout << "number of threads = " << num_threads << std::endl;

  for(int i = 6; i <= 11; ++i){
    const unsigned int size = 1 << i;

    double * A = (double *) malloc (sizeof(double) * size * size);
    double * B = (double *) malloc (sizeof(double) * size * size);
    double * C = (double *) malloc (sizeof(double) * size * size);

    for( unsigned int i = 0; i < size; i++){
      for( unsigned int j = 0 ; j < size ; j++){
        A[i*size+j] = i;
        B[i*size+j] = j;
      }
    }

    // start time measurement
    auto start = std::chrono::steady_clock::now();

    MatrixMultiplicationSerial(A, B, C, size);

    auto t1 = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now()-start).count();

    auto start2 = std::chrono::steady_clock::now();

    MatrixMultiplicationParallel( A, B, C, size, num_threads);

    auto t2 = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now()-start2).count();

    std::cout << "\nMatrix size: " << size << "\nserial time: "<< t1 << "ms"<<"\nparallel time:"
              << t2 << "ms" << std::endl;

    free(A);
    free(B);
    free(C);
  }


  return 0;
}
