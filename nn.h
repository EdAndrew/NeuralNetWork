#ifndef _NEURAL_NETWORK_H
#define _NEURAL_NETWORK_H
typedef struct NeuralNetWork
{
	int d;	//输入层个数
	int q;	//隐层个数
	int l;  //输出层个数
	double **wh, **wo;
	double rate;
	
}NNet;

#endif
