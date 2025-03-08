#include <a1.hpp>
#include <vector>
#include <thread>
#include <cstring>


/** \brief The namespace of the first assignment
*/
namespace a1{
  /**\brief A serial implementation of a matrix-matrix multiplication C=A*B for square matrices.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix that is the left hand side of the multiplication.
  * @param B a row-major order matrix that is the right hand side of the multiplication
  * @param C a row-major order matrix that is the result of the multiplication
  * @param size the size of one dimension for the square matrices.
  */
  void MatrixMultiplicationSerial(const double* A, const double* B, double* C, const unsigned int size){
    memset(C, 0, sizeof(double)*size*size);

    for(unsigned int i = 0; i < size; ++i){
      for(unsigned int j = 0; j < size; ++j){
        for(unsigned int k = 0; k < size; ++k){
          C[i*size + k] += A[i*size + j] * B[j*size + k];
        }
      }
    }
  }

  /**\brief A worker function for the parallel matrix-matrix multiplication.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  * This function does not use any synchronization primitives.
  *
  * @param A a row-major order matrix that is the left hand side of the multiplication.
  * @param B a row-major order matrix that is the right hand side of the multiplication
  * @param C a row-major order matrix that is the result of the multiplication
  * @param size the size of one dimension for the square matrices.
  * @param tid the thread id
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationWorker(const double* A, const double* B, double* C, const unsigned int size, const unsigned int tid, const unsigned int num_threads){
    unsigned int chunk_size = size / num_threads + (size%num_threads == 0 ? 0 : 1); //size*size / num_threads; ??  
    unsigned int start = tid * chunk_size;
    unsigned int end = std::min(size, (tid+1)*chunk_size);

    for(unsigned int i = start; i < end; ++i){
      for(unsigned int j = 0; j < size; ++j){
        for(unsigned int k = 0; k < size; ++k){
          C[i*size + k] += A[i*size + j] * B[j*size + k];
        }
      }
    }
  }

  /**\brief A parallel implementation of a matrix-matrix multiplication C=A*B for square matrices.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix that is the left hand side of the multiplication.
  * @param B a row-major order matrix that is the right hand side of the multiplication
  * @param C a row-major order matrix that is the result of the multiplication
  * @param size the size of one dimension for the square matrices.
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationParallel(const double* A, const double* B, double* C, const unsigned int size, const unsigned int num_threads){
    memset(C, 0, sizeof(double)*size*size);
    void (*f)(const double*, const double*, double*, const unsigned int, const unsigned int, const unsigned int) = MatrixMultiplicationWorker;

    std::vector<std::thread> th;
    for(unsigned int tid = 0; tid < num_threads; ++tid){
      th.push_back( std::thread(f, A, B, C, size, tid, num_threads));
    }

    for(unsigned int tid = 0; tid < num_threads; ++tid){
      th[tid].join();
    }
  }

////////////////////////////////////////////////////////////////////////////////
/// BONUS
////////////////////////////////////////////////////////////////////////////////

  /**\brief A serial implementation of a matrix-matrix multiplication C=A*B.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix of dimensions N*K (N rows, K cols)
  * @param B a row-major order matrix of dimensions K*M (K rows, M cols)
  * @param C the resulting row-major order matrix of dimensions N*M (N rows, M cols)
  * @param N number of rows of A and C
  * @param K number of cols of A and rows of B
  * @param M number of cols of B and C
  */
  void MatrixMultiplicationSerial(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M){
    memset(C, 0, sizeof(double)*N*M);

    for(unsigned int i = 0; i < N; ++i){
     for(unsigned int j = 0; j < K; ++j){
       for(unsigned int k = 0; k < M; ++k){
          C[i*M + k] += A[i*K + j] * B[j*M + k];
        }
      }
    }
  }

  /**\brief A worker function for the parallel matrix-matrix multiplication.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  * This function does not use any synchronization primitives.
  *
  * @param A a row-major order matrix of dimensions N*K (N rows, K cols)
  * @param B a row-major order matrix of dimensions K*M (K rows, M cols)
  * @param C the resulting row-major order matrix of dimensions N*M (N rows, M cols)
  * @param N number of rows of A and C
  * @param K number of cols of A and rows of B
  * @param M number of cols of B and C
  * @param tid the thread id
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationWorker(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M,const unsigned int tid, const unsigned int num_threads){
    unsigned int chunk_size = (N+num_threads-1) / num_threads; 
    unsigned int start = tid * chunk_size;
    unsigned int end = std::min(N, (tid+1)*chunk_size);

    for(unsigned int i = start; i < end; ++i){
     for(unsigned int j = 0; j < K; ++j){
       for(unsigned int k = 0; k < M; ++k){
          C[i*M + k] += A[i*K + j] * B[j*M + k];
        }
      }
    }
  }

  /**\brief A parallel implementation of a matrix-matrix multiplication C=A*B.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix of dimensions N*K (N rows, K cols)
  * @param B a row-major order matrix of dimensions K*M (K rows, M cols)
  * @param C the resulting row-major order matrix of dimensions N*M (N rows, M cols)
  * @param N number of rows of A and C
  * @param K number of cols of A and rows of B
  * @param M number of cols of B and C
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationParallel(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int num_threads){
    memset(C, 0, sizeof(double)*N*M);
    void (*f)(const double*, const double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int) = MatrixMultiplicationWorker;

    std::vector<std::thread> th;
    for(unsigned int tid = 0; tid < num_threads; ++tid){
      th.push_back(std::thread(f, A, B, C, N, K, M, tid, num_threads));
    }
    for(unsigned int tid = 0; tid < num_threads; ++tid){
      th[tid].join();
    }
  }
  

}
