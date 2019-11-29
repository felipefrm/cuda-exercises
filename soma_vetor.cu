#include <stdio.h>
#include <stdlib.h>

#define N 32

//Código device
__global__ void soma_vetor(int *a, int *b, int *c){
	int indice = blockIdx.x;
	if(indice < N)
		c[indice] = a[indice] + b[indice];
}

//Código host
int main(){
	int a[N],b[N],c[N];
	int* dev_a;
	int* dev_b;
	int* dev_c;

	int tam = N*sizeof(int);
	//Inicializando as variáveis do host:
	for(int i = 0; i < N; i++){
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

	soma_vetor<<<N,1>>>(dev_a, dev_b, dev_c);

	//Copiando o resultado da GPU para a CPU:
	cudaMemcpy(&c, dev_c, tam, cudaMemcpyDeviceToHost);

	//Visualizando o resultado:
	for(int i = 0; i < N; i++)
		printf("%d ",c[i]);
	printf("\n\n");

	//Liberando a memória na GPU:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
