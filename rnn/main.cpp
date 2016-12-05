#include <iostream>

#include "network.h"
#include "weights.h"
#include "ulti.h"


int main()
{
	std::vector<LayerInfo> net_info;
	size_t NUM_LAYERS;
	NUM_LAYERS = sizeof(SIZE) / sizeof(SIZE[0]);
	for (size_t i = 0; i < NUM_LAYERS - 1; i++)
		net_info.push_back(LayerInfo(tanh, SIZE[i]));
	net_info.push_back(LayerInfo(sigmoid, SIZE[NUM_LAYERS-1]));
	std::vector<std::vector<std::vector<double> > > Synapses = arr2vec3d(WEIGHTS, SIZE, NUM_LAYERS);
	std::vector<std::vector<double> > Biases = arr2vec2d(BIASES, SIZE, NUM_LAYERS);
	
	//std::cout<<"###capacity##"<<Synapses[3][0][65]<<std::endl;
	
	std::cout<<"#######"<<Synapses[1][0].size()<<std::endl;
	for (size_t i = 0; i < Synapses[1][0].size(); i++)
		std::cout<<Synapses[1][0][i] <<std::endl;
	Net test_net(net_info, Synapses, Biases);
	
	
	std::vector<double> Inputs;
	for (size_t i = 0; i < SIZE[0]; i++)
	{
		Inputs.push_back(1);
	}
	//std::cout<<Inputs.size()<<std::endl;
	test_net.setInputVals(Inputs);
	for (int i = 0; i < 40; i++)
		test_net.feedForward();
	std::cout<<"output: "<<test_net.getOutputVals()[0]<<std::endl;

	
	return 0;
}
