#include <stdio.h>
#include <stdlib.h>

#define N 32 // quantidades de numeros
#define I 32  // intervalo

// codigo device
__global__ void soma_numeros(int *a, int *res) {

  __shared__ int temp[N/I];

  int ind = threadIdx.x;
 
  if (ind < N/I){
    int soma = 0;
    for (int i = I*ind; i < I*ind + I; i++)
      soma += a[i];
    temp[ind] = soma;
  }

  __syncthreads();

  int resto;
  int controle = I;
  int somaThreads = 0;
 
  while (controle <= N) {
    if (ind < N/controle/I) {
      for (int i = I*ind; i<I*ind+I; i++)
        somaThreads += temp[i];
    temp[ind] = somaThreads;
    somaThreads = 0;
    }
    
    resto=N/controle;
    if (resto == 0)
      resto = 1;
    controle *= I;
}
  
  __syncthreads();
  
  if (ind == 0){
    printf("%d", resto);
    *res = 0;
    for (int i = 0; i < resto; i++) {
      *res += temp[i];
    }
  }
}

// Código host
int main(){
  int a[N];
  int r;
  int* dev_a;
  int* dev_r;

  // Inicializando as variaveis do host
  for (int i = 0; i < N; i++)
    a[i] = i;

  // Alocando espaço para as variaveis da GPU
  cudaMalloc((void**)&dev_a, N*sizeof(int));
  cudaMalloc((void**)&dev_r, sizeof(int));

  // Copiando as variaveis da CPU para a GPU
  cudaMemcpy(dev_a, &a, N*sizeof(int), cudaMemcpyHostToDevice);

  // Chamada à função da GPU (kernel)
  // Número de blocos é igual à dimensão do vetor dividida pela dimensão do bloco: N/M

  // O tipo dim3 permite definir a quantidade de blocos e threads por dimensao.
  // A terceira dimensao é omitida, ficando implícito o valor 1.
  soma_numeros<<<1, N/I>>>(dev_a, dev_r);

  // Copiando o resultado da GPU para CPU
  cudaMemcpy(&r, dev_r, sizeof(int), cudaMemcpyDeviceToHost);

  // Visualizando o resultado
  printf("R = %d\n", r);

  // Liberando a memoria na GPU
  cudaFree(dev_a);
  cudaFree(dev_r);

  return 0;
}
