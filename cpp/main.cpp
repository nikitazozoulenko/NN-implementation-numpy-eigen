#include <Eigen/Dense>
#include <string>
#include <vector>

#include <iostream>
#include "NeuralNetwork.h"
#include "Layer.h"

#include "ProductInputCreator.h"

using Eigen::Matrix;
using Eigen::MatrixXd;

int main()
{
	Layer layer_input(
		"input",	//type
		0,			//index
		2);			//neurons

	Layer layer_hidden1(
		"hidden1",	//type
		1,			//index
		3);		//neurons

	Layer layer_hidden2(
		"hidden2",	//type
		2,			//index
		3);		//neurons

	Layer layer_hidden3(
		"hidden3",	//type
		3,			//index
		3);		//neurons

	Layer layer_output(
		"output",	//type
		4,			//index
		1);			//neurons

	int batch_size = 3;

	std::vector<Layer> layers;
	layers = { layer_input,
			   layer_hidden1,
			   layer_hidden2,
			   layer_hidden3,
			   layer_output };

	NeuralNetwork network(layers, batch_size);

	ProductInputCreator creator(batch_size);

	MatrixXd in(3, 2);
	in << 1.0, 1.0,
		1.0, 0.5,
		0.5, 0.5;

	MatrixXd o(3, 1);
	o << 1.0,
		0.5,
		0.25;

	network.forward(in);
	network.backprop(o);
	network.print_forward();
	network.print_backprop();

	std::cout << std::endl << std::endl << std::endl << std::endl;
	for (int i = 0; i != 100000; ++i)
	{
		creator.gen_values();
		MatrixXd in = creator.get_input();
		MatrixXd o = creator.get_output();
		network.forward(in);
		network.backprop(o);

	}

	network.forward(in);
	network.backprop(o);
	network.print_forward();
	network.print_backprop();

	std::cin.get();

	return 0;
}