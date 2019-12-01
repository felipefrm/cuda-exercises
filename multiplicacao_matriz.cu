#include <stdio.h>
#include <stdlib.h>

#define N 3 
#define M 3  
#define K 3
#define T 8 // numero de threads por bloco

// codigo device
__global__ void multiplica_matriz(int *a, int *b, int *c){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  c[N*i+j] = 0;
  if (i < M && j < N)
    for (int p = 0; p<K; p++){
      c[N*i+j] += a[i*K+p] * b[N*p+j];
    }
}

// Código host
int main(){
  int a[M*K], b[K*N], c[M*N];
  int* dev_a;
  int* dev_b;
  int* dev_c;

  // Inicializando as variaveis do host
  for (int i = 0; i < M*K; i++){
    a[i] = i;
  for (int i = 0; i < K*N; i++)
    b[i] = i*2;
  }

  // Alocando espaço para as variaveis da GPU
  cudaMalloc((void**)&dev_a, M*K*sizeof(int));
  cudaMalloc((void**)&dev_b, K*N*sizeof(int));
  cudaMalloc((void**)&dev_c, M*N*sizeof(int));

  // Copiando as variaveis da CPU para a GPU
  cudaMemcpy(dev_a, &a, M*K*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, &b, K*N*sizeof(int), cudaMemcpyHostToDevice);

  // Chamada à função da GPU (kernel)
  // Número de blocos é igual à dimensão do vetor dividida pela dimensão do bloco: N/M

  // O tipo dim3 permite definir a quantidade de blocos e threads por dimensao.
  // A terceira dimensao é omitida, ficando implícito o valor 1.
  dim3 numBlocos((M+T-1)/T,(N+T-1)/T);
  dim3 numThreads(T,T); 
  multiplica_matriz<<<numBlocos, numThreads>>>(dev_a, dev_b, dev_c);

  // Copiando o resultado da GPU para CPU
  cudaMemcpy(&c, dev_c, M*N*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < M; i++){
    for (int j = 0; j < K; j++)
      printf("%d ", a[K*i+j]);
    printf("\n");
  }
 
 print("X");
 
 for (int i = 0; i < K; i++){
    for (int j = 0; j < N; j++)
      printf("%d ", b[N*i+j]);
    printf("\n");
  }
 
 
  // Visualizando o resultado
  for (int i = 0; i < M; i++){
    for (int j = 0; j < N; j++)
      printf("%d ", c[N*i+j]);
    printf("\n");
  }
  // Liberando a memoria na GPU
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}

// 4 threads/bloco -> indice = 4*IdBloco + IdThread
//  numero de blocos = (tamanho vetor / dimensão do bloco) -> funciona se os dois valores sao multiplos
//  numero de blocos = ((tamanho vetor + dimensao do bloco -1) / dimensao do bloco) -> funciona sempre!
