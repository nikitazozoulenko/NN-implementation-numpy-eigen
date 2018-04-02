#ifndef NEURAL_NET_LAYER_H
#define NEURAL_NET_LAYER_H

#include <string>

class Layer
{
public:
	Layer(std::string type, unsigned index, int neurons) : type(type), index(index), neurons(neurons) {}

	std::string type;
	unsigned index;
	int neurons;
protected:
private:
};

#endif //NEURAL_NET_LAYER_H