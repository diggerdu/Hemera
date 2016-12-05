#include "network.h"
//#include <iostream>
LayerInfo::LayerInfo(double (*activation_function )(double sum), size_t NumNode)
{
    activate = activation_function;
    numnodes = NumNode;
};


Node::Node(const std::vector<double>& Synapse, const double m_bias, double(*activation_function)(double sum))
{
    activate = activation_function;
    synapse = Synapse;
    bias = m_bias;
    OutputVal = 0;
};

void Node::init()
{
    setOutputVal(0);
};


void Node::feedForward(const std::vector<double>& InputVals)
{

    double sum = 0;
	assert(InputVals.size() == synapse.size());
    for (unsigned i = 0; i < synapse.size(); i++)
	{
        sum += InputVals[i] * synapse[i];
	}
	
	
	sum += bias;
//	if (synapse.size() == 90)
//		if (sum > 100)
//				std::cout<<"sumbug"<<synapse[0]<<std::endl;
    setOutputVal(activate(sum));
};



Layer::Layer(const std::vector<std::vector<double> >& Synapses, const std::vector<double>& Biases, size_t NumNode, double(*activation_function)(double sum))
{
        for (unsigned i = 0; i < NumNode; i++)
            nodes.push_back(Node(Synapses[i], Biases[i], activation_function));
};


void Layer::init()
{
    for (size_t i = 0; i < nodes.size(); i++)
        nodes[i].init();
};

void Layer::setOutputVals(std::vector<double> inputVals)
{
    assert(nodes.size() == inputVals.size());
    for (size_t i = 0; i < inputVals.size(); i++)
        nodes[i].setOutputVal(inputVals[i]);
};

std::vector<double> Layer::getOutputVals()
{
    std::vector<double> OutputVals;
    for (unsigned i = 0; i < nodes.size(); i++)
        OutputVals.push_back(nodes[i].getOutputVal());
    return OutputVals;
};

void Layer::feedForward(Layer& PrevLayer)
{
    std::vector<double> PrevOutputs;
    std::vector<double> CurrentOutputs;
    std::vector<double> CurrentInputs;
    PrevOutputs = PrevLayer.getOutputVals();
    CurrentOutputs = this->getOutputVals();
    CurrentInputs.reserve(PrevOutputs.size() + CurrentOutputs.size());
    CurrentInputs.insert(CurrentInputs.end(), PrevOutputs.begin(), PrevOutputs.end());
    CurrentInputs.insert(CurrentInputs.end(), CurrentOutputs.begin(), CurrentOutputs.end());
//	if (CurrentInputs.size() == 90)
//		for (size_t i = 0; i < 90; i++)
//			std::cout<<"Inputs"<<CurrentInputs[i]<<std::endl;
	for (unsigned i = 0; i < nodes.size(); i++)
 	       nodes[i].feedForward(CurrentInputs);
};



Net::Net(const std::vector<LayerInfo>& Topo, const std::vector<std::vector<std::vector<double> > >& Synapses, const std::vector<std::vector<double> > Biases) 
{
    for (unsigned i = 0; i < Topo.size(); i++)
        layers.push_back(Layer(Synapses[i], Biases[i], Topo[i].numnodes, Topo[i].activate));
};

void Net::init()
{
    for (unsigned i = 0; i < layers.size(); i++)
        layers[i].init();
};
void Net::setInputVals(const std::vector<double> & InputVals)
{
    assert(layers.size() > 1);
    layers[0].setOutputVals(InputVals);
};

void Net::feedForward()
{
	//std::cout<<"SIZE"<<layers.size()<<std::endl;
    assert(layers.size() > 1);
    for (unsigned i = 1; i < layers.size(); i++)
        layers[i].feedForward(layers[i-1]);
};

std::vector<double> Net::getOutputVals()
{
    return layers[layers.size()-1].getOutputVals();
};

/*range from 0 to 1
 */
extern double sigmoid(double x)
{
    return 1/(1+exp(-1*x));
}


/*range from -1 to 1
 */
extern double sigmoid_h(double x)
{
    return 2/(1+exp(-x))-1;
}

/* range from -2 to 2
 */
extern double sigmoid_g(double x)
{
    return 4/(1+exp(-x))-2;
}

extern double relu(double x)
{
    return x > 0 ? x : 0;
}

extern double tanh(double x)
{
    return (exp(2*x) - 1) / (exp(2*x) + 1);
}
