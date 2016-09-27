#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

int main()
{
	NNet nn;
	NNet_init(&nn, 3, 4, 3);
	NNet_fini(&nn);
}






