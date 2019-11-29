#include <stdio.h>
#include <stdlib.h>

#define N 2 //Número de colunas das matrizes
#define M 2  //Número de linhas das matrizes
#define T 8  //Número de threads por bloco

//Código device
// blockDim.x é a dimensão do bloco, ou seja,
// a quantidade de threads por bloco.

__global__ void soma_matriz(int *a, int *b, int *c){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i < M && j < N)
		c[N*i + j] = a[N*i + j] + b[N*i + j];
}

//Código host
int main(){
	int a[M*N],b[M*N],c[M*N];
	int* dev_a;
	int* dev_b;
	int* dev_c;

	int tam = M*N*sizeof(int);
	//Inicializando as variáveis do host:
	for(int i = 0; i < M*N; i++){
		a[i] = i;
		b[i] = i*2;
	}
	
	//Alocando espaço para as variáveis da GPU:
	cudaMalloc((void**)&dev_a, tam);
	cudaMalloc((void**)&dev_b, tam);
	cudaMalloc((void**)&dev_c, tam);
	
	//Copiando as variáveis da CPU para a GPU:
	cudaMemcpy(dev_a, &a, tam, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, tam, cudaMemcpyHostToDevice);

	//Chamada à função da GPU (kernel):
	//Número de blocos é igual à dimensão do vetor
	//dividida pela dimensão do bloco: N/M

	// O tipo dim3 permite definir a quantidade de
	// blocos e threads por dimensão. A terceira dimensão é omitida,
	// ficando implícito o valor 1.
	dim3 numBlocos (2,2); // número de blocos = 2x2 = 4
	dim3 numThreads (2,2); // número de threads por bloco = 2x2 = 4
	soma_matriz<<<numBlocos,numThreads>>>(dev_a, dev_b, dev_c);

	//Copiando o resultado da GPU para a CPU:
	cudaMemcpy(&c, dev_c, tam, cudaMemcpyDeviceToHost);

	//Visualizando o resultado:
	for(int i = 0; i < N; i++){
		for(int j = 0; j < M; j++)
			printf("%d ",c[N*i+j]);
		printf("\n");
	}
	printf("\n\n");

	//Liberando a memória na GPU:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
