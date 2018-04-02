#ifndef PRODUCTINPUTCREATOR
#define PRODUCTINPUTCREATOR

#include "NeuralNetwork.h"

class ProductInputCreator
{
public:
	ProductInputCreator(int batch_size);
	MatrixXd get_input();
	MatrixXd get_output();
	void gen_values();
protected:
private:
	int batch_size;
	MatrixXd input;
	MatrixXd output;
};
#endif // !PRODUCTINPUTCREATOR
