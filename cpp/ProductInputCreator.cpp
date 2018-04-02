#include "ProductInputCreator.h"

ProductInputCreator::ProductInputCreator(int batch_size) : batch_size(batch_size)
{

}

void ProductInputCreator::gen_values()
{
	input = MatrixXd::Random(batch_size, 2);
	output = input.col(0).cwiseProduct(input.col(1));
}

MatrixXd ProductInputCreator::get_input()
{
	return input;
}

MatrixXd ProductInputCreator::get_output()
{
	return output;
}