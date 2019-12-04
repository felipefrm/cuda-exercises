#include <stdio.h>
#include <stdlib.h>

#define N 4 // quantidades de numeros
#define I 2 // adjacentes

// codigo device
__global__ void soma_adj(int *a, int *b){

    __shared__ int temp[N];
    int ind = threadIdx.x;
    int pos_inicio = ind - I;
    int pos_final = ind + I +1;

    if (ind < N){
      int soma = 0;
      for (int i=pos_inicio; i<pos_final; i++){
        if (i >= 0 && i < N)
          soma += a[i];
      }
      temp[ind] = soma;
    }

    b[ind] = temp[ind];
}




// Código host
int main(){
  int a[N];
  int b[N];
  int* dev_a;
  int* dev_b;

  // Inicializando as variaveis do host
  for (int i = 0; i < N; i++)
    a[i] = i+1;

  // Alocando espaço para as variaveis da GPU
  cudaMalloc((void**)&dev_a, N*sizeof(int));
  cudaMalloc((void**)&dev_b, N*sizeof(int));

  // Copiando as variaveis da CPU para a GPU
  cudaMemcpy(dev_a, &a, N*sizeof(int), cudaMemcpyHostToDevice);

  // Chamada à função da GPU (kernel)
  // A terceira dimensao é omitida, ficando implícito o valor 1.
  soma_adj<<<1, N>>>(dev_a, dev_b);

  // Copiando o resultado da GPU para CPU
  cudaMemcpy(&b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost);

  // Visualizando o resultado
  for (int i=0; i<N; i++)
    printf("%d \n", b[i]);

  // Liberando a memoria na GPU
  cudaFree(dev_a);
  cudaFree(dev_b);

  return 0;
}
