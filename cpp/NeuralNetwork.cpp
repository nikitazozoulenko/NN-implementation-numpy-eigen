#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <string>


NeuralNetwork::NeuralNetwork(std::vector<Layer> layers, int batch_size) : layers(layers), batch_size(batch_size)
{
	//"init" the matrices
	for (int i = 0; i != layers.size()-1; ++i)
	{
		A_vector.push_back(MatrixXd(batch_size, layers[i].neurons));
		Z_vector.push_back(MatrixXd(batch_size, layers[i].neurons));

		W_vector.push_back(MatrixXd::Random(layers[i].neurons, layers[i + 1].neurons));
		dKdW_vector.push_back(MatrixXd(layers[i].neurons, layers[i + 1].neurons));
	}

	//Y_hat
	A_vector.push_back(MatrixXd(batch_size, layers[layers.size() - 1].neurons));

	//scale factor for training
	scale_factor = 1.0;
}

void NeuralNetwork::forward(MatrixXd &X)
{
	A_vector[0] = X;

	for (int i = 0; i != layers.size() - 1; ++i)
	{
		Z_vector[i] = A_vector[i] * W_vector[i];
		A_vector[i + 1] = Z_vector[i].unaryExpr(&(NeuralNetwork::sigmoid));
	}
}

void NeuralNetwork::backprop(MatrixXd &Y)
{
	MatrixXd Y_hat = get_Y_hat();
	MatrixXd delta;

	for (int i = dKdW_vector.size() - 1; i != -1; --i)
	{
		if (i == dKdW_vector.size() - 1)
		{
			delta = (Y - Y_hat).cwiseProduct(Z_vector[i].unaryExpr(&(NeuralNetwork::sigmoid_prime)));
		}
		else
		{
			delta = (delta * W_vector[i + 1].transpose()).cwiseProduct(Z_vector[i].unaryExpr(&(NeuralNetwork::sigmoid_prime)));
		}

		dKdW_vector[i] = A_vector[i].transpose() * delta;		//ÄNDAR VIKTERNA PÅ DIREKTEN
		//W_vector[1] += scale_factor * dKdW_vector[1];         //ELLER DET ANDRA ALTERNATIVET NEDAN:

	}

	for (int i = dKdW_vector.size() - 1; i != -1; --i)
	{
		W_vector[i] += scale_factor * dKdW_vector[i];           //ÄNDRA ALLA VIKTER SAMTIDIGT
	}
}

double NeuralNetwork::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoid_prime(double x)
{
	return 1.0 * exp(-x) / ((1.0 + exp(-x)) * (1.0 + exp(-x)));
}

MatrixXd NeuralNetwork::get_Y_hat()
{
	return A_vector[layers.size() - 1];
}

void NeuralNetwork::print(MatrixXd m, std::string name)
{
	std::cout << name << " =\n" << m << std::endl << std::endl;
}

void NeuralNetwork::print_forward()
{
	print(A_vector[0], "X");
}

void NeuralNetwork::print_backprop()
{
	print(get_Y_hat(), "Y_hat");
}
