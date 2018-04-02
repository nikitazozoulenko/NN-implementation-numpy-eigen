#ifndef NEURAL_NETWORK_V1_H
#define NEURAL_NETWORK_V1_H

#include <Eigen/Dense>
#include "Layer.h"
#include <vector>

using Eigen::Matrix;
using Eigen::MatrixXd;

class NeuralNetwork
{
public:

	NeuralNetwork(std::vector<Layer> layers, int batch_size);

	void forward(MatrixXd &X);
	void backprop(MatrixXd &Y);

	void print_forward();
	void print_backprop();

	MatrixXd get_Y_hat();

	void print(MatrixXd m, std::string name);

protected:
private:
	static double sigmoid(double x);
	static double sigmoid_prime(double x);

	std::vector<Layer> layers;
	int batch_size;
	double scale_factor;

	std::vector<MatrixXd> Z_vector;
	std::vector<MatrixXd> A_vector;
	std::vector<MatrixXd> W_vector;
	std::vector<MatrixXd> dKdW_vector;
};

#endif //NEURAL_NETWORK_V1_H
