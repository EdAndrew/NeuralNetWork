#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"

double rand_0_1()
{
	static int flag = 0;
	if (!flag)
	{
		srand(time(NULL));
		flag = 1;
	}

	return (rand() % 10) * 0.1;
}

double sigmoid(double x)
{
	return 1 / (exp(-x) + 1);
}

int NNet_init(NNet *pN, int d, int q, int l)
{
	if (pN == NULL)
	{
		fprintf(stderr, "Neural network doesn't exist.\n");
		return -1;
	}
	if (d <= 0|| q <= 0|| l <= 0)
	{
		fprintf(stderr, "neural network parameter error.\n");
		return -1;
	}
	pN->d = d;
	pN->q = q;
	pN->l = l;
	
	int i, h, j, t, i1;
	
	//初始化隐层参数
	pN->wh = malloc(q * sizeof(double *));
	if (pN->wh == NULL)
	{
		fprintf(stderr, "Out of memory\n");
		return -1;
	}
	for (i1 = 0; i1 < q; ++i1)
	{
		pN->wh[i1] = malloc((d+1) * sizeof(double));
		if (pN->wh[i1] == NULL)
		{
			fprintf(stderr, "Out of memory\n");
			goto e1;
		}
	}
	for (h = 0; h < q; ++h)
		for (i = 0; i < d+1; ++i)
	{
		pN->wh[h][i] = rand_0_1();
		//printf("wh[%d][%d] = %f\n", h, i, pN->wh[h][i]);
	}
	
	//初始化输出层参数
	pN->wo = malloc(l * sizeof(double *));
	if (pN->wo == NULL)
	{
		fprintf(stderr, "Out of memory\n");
		goto e1;
	}
	for (i = 0; i < l; ++i)
	{
		pN->wo[i] = malloc((q+1) * sizeof(double));
		if (pN->wo[i] == NULL)
		{
			fprintf(stderr, "Out of memory\n");
			goto e2;
		}
	}
	for (j = 0; j < l; ++j)
		for (h = 0; h < q+1; ++h)
	{
		pN->wo[j][h] = rand_0_1();
		//printf("wo[%d][%d] = %f\n", j, h, pN->wo[j][h]);
	}

	return 0;

e2:
	for (t = 0; t < i; ++t)
		free(pN->wo[t]);
	free(pN->wo);
e1:
	for (t = 0; t < i1; ++t)
		free(pN->wh[t]);	
	free(pN->wh);
	return -1;	
}

int NNet_fini(NNet *pN)
{
	if (pN == NULL)
		return -1;
	
	int i;

	if (pN->wo != NULL)
	{
		for (i = 0; i < pN->l; ++i)
		{
			if (pN->wo[i] != NULL)
			{
				free(pN->wo[i]);
				pN->wo[i] = NULL;
			}
		}
	}
	
	if (pN->wh != NULL)
	{
		for (i = 0; i < pN->q; ++i)
		{
			if (pN->wh[i] != NULL)
			{
				free(pN->wh[i]);
				pN->wh[i] = NULL;
			}
		}
	}

	return 0;
}

int do_predict(NNet *pN, double *input, double *output, double *outHidden)
{
	if (pN == NULL)
	{
		fprintf(stderr, "Network is not exist.\n");
		return -1;
	}
	if (input == NULL)
	{
		fprintf(stderr, "data is not exist.\n");
		return -1;
	}

	int i, j;	
	double outTmp;
	
	//Compute the output of hidden layer
	for (i = 0; i < pN->q; ++i)
	{
		outTmp = 0.0;
		for (j = 0; j < pN->d; ++j)
		{
			outTmp += input[j] * pN->wh[i][j];
		}
		outTmp += 1.0 * pN->wh[i][pN->d];
		outHidden[i] = sigmoid(outTmp);	
		
	}

	//Compute the output of output layer
	for (i = 0; i < pN->l; ++i)
	{
		outTmp = 0.0;
		for (j = 0; j < pN->q; ++j)
		{
			outTmp += outHidden[i] * pN->wo[i][j];
		}
		outTmp += 1.0 * pN->wo[i][pN->q];
		output[i] = sigmoid(outTmp);
	}
	
	return 0;	
}

int inc_gred_out(double *output, int len, int *target, double *gradOut)
{		
	int i;
	for (i = 0; i < len; ++i)
	{
		gradOut[i] = output[i] * (1 - output[i]) * ((double)target[i] - output[i]);
	}
	return 0;
}

int inc_gred_hidden(NNet *pN, double *outHidden, double *gradOut, double *gradHidden)
{
	int i, j;
	double sum;
	for (i = 0; i < pN->q; ++i)
	{
		sum = 0.0;
		for (j = 0; j < pN->l; ++j)
		{
			sum += pN->wo[j][i] * gradOut[j];
		}
		gradHidden[i] = sum * (outHidden[i]) * (1 - outHidden[i]);
	}
	return 0;
}


int NNet_train(NNet *pN, double **train, int *target, int size, double rate)
{
	if (pN == NULL)
	{
		fprintf(stderr, "network is not exist.\n");
		return -1;
	}
	if (train == NULL)
	{
		fprintf(stderr, "train is not exist.\n");
		return -1;
	}
	if (target == NULL)
	{
		fprintf(stderr, "target is not exist.\n");
		return -1;
	}
	if (size <= 0)
	{
		fprintf(stderr, "size is error.\n");
		return -1;
	}
	if (rate <= 1e-15 || rate >= 1.0 - 1e-15)
	{
		fprintf(stderr, "rate is error.\n");
		return -1;
	}

	int i, j, k;
	double output[pN->l];
	double outHidden[pN->q];
	double gradOut[pN->l], gradHidden[pN->q];
	double deltaOut, deltaHidden;
	double e;
	do
	{
		for (i = 0; i < size; ++i)
		{
			do_predict(pN, train[i], output, outHidden);
			inc_gred_out(output, pN->l, target, gradOut);
			inc_gred_hidden(gradHidden);
			
			//update weight of out layer 
			for (j = 0; j < pN->l; ++j)
			{
				for (k = 0; k < pN->q; ++k)
				{
					pN->wo[j][k] += rate * gradOut[j] * outHidden[k];
				}
			}
			for (j = 0; j < pN->l; ++j)
			{
				pN->wo[j][pN->l] += -1.0 * rate * gradeOut[j];
			}
			

			//update weight of hidden layer
			for (j = 0; j < pN->q; ++j)
			{
				for(k = 0; k < pN->d; ++k)
				{
					pN->wh[j][k] += rate * gradHidden[j] * input[k];
				}
			}
			for (j = 0; j < pN->l; ++j)
			{
				pN->wh[j][pN->d] += -1.0 * rate * gradHidden[j];
			}
			
		}

	}while(E < MIN_TRAIN ||i < MAX_TRAIN);	
		

}

int main()
{
	NNet nn;
	int i;
	double input[3] = {1, 1, 1};
	double output[3];

	NNet_init(&nn, 3, 4, 3);
	do_predict(&nn, input, output);
	for (i = 0; i < 3; ++i)
		printf("%f ", output[i]);
	printf("\n");
	NNet_fini(&nn);
}











