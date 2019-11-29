#include <stdio.h>
#include <stdlib.h>

#define N 3 // numero de colunas das matrizes
#define M 3  // número de threads por bloco
#define T 8 // numero de threads por bloco

// codigo device
__global__ void multiplica_matriz(int *a, int *b, int *c){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  c[N*i+j] = 0;
  if (i < M && j < N)
    for (int k = 0; k<N; k++){
      c[N*j+i] += a[j*N+k] * b[N*k+i];
    }
}

// Código host
int main(){
  int a[M*N], b[M*N], c[M*N];
  int* dev_a;
  int* dev_b;
  int* dev_c;

  // Inicializando as variaveis do host
  for (int i = 0; i < N*M; i++){
    a[i] = i;
    b[i] = i*2;
  }

  // Alocando espaço para as variaveis da GPU
  int tam = N*M*sizeof(int);
  cudaMalloc((void**)&dev_a, tam);
  cudaMalloc((void**)&dev_b, tam);
  cudaMalloc((void**)&dev_c, tam);

  // Copiando as variaveis da CPU para a GPU
  cudaMemcpy(dev_a, &a, tam, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, &b, tam, cudaMemcpyHostToDevice);

  // Chamada à função da GPU (kernel)
  // Número de blocos é igual à dimensão do vetor dividida pela dimensão do bloco: N/M

  // O tipo dim3 permite definir a quantidade de blocos e threads por dimensao.
  // A terceira dimensao é omitida, ficando implícito o valor 1.
  dim3 numBlocos((M+T-1)/T,(N+T-1)/T);
  dim3 numThreads(T,T); 
  multiplica_matriz<<<numBlocos, numThreads>>>(dev_a, dev_b, dev_c);

  // Copiando o resultado da GPU para CPU
  cudaMemcpy(&c, dev_c, tam, cudaMemcpyDeviceToHost);

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
