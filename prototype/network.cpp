#include "network_pre.h"




LayerInfo::LayerInfo::LayerInfo(double (*activation_function )(double sum), size_t NumNode)
{
    activate = activation_function;
    numnodes = NumNode;
};


Node::Node(const std::vector<double>& Synapse, double(*activation_function)(double sum)):activate(activation_function)
{
    synapse = Synapse;
    OutputVal = 0;
};

void Node::feedForward(const std::vector<double>& InputVals)
{
    double sum = 0;
    assert(InputVals.size() == synapse.size());
    for (unsigned i = 0; i < synapse.size(); i++)
        sum += InputVals[i] * synapse[i];
    setOutputVal(activate(sum));
};




Layer::Layer(const std::vector<std::vector<double> >& Synapses, size_t NumNode, double(*activation_function)(double sum))
{
        assert(Synapses.size() == NumNode);
        for (unsigned i = 0; i < NumNode; i++)
            nodes.push_back(Node(Synapses[i], activation_function));
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
    PrevOutputs = PrevLayer.getOutputVals();
    for (unsigned i = 0; i < nodes.size(); i++)
        nodes[i].feedForward(PrevOutputs);
};



Net::Net(const std::vector<LayerInfo>& Topo, const std::vector<std::vector<std::vector<double> > >& Synapses)
{
    for (unsigned i = 0; i < Topo.size(); i++)
        layers.push_back(Layer(Synapses[i], Topo[i].numnodes, Topo[i].activate));
};

void Net::setInputVals(const std::vector<double> & InputVals)
{
    assert(layers.size()>1);
    layers[1].setOutputVals(InputVals);
};

void Net::feedForward()
{
    assert(layers.size() > 1);
    for (unsigned i = 1; i < layers.size(); i++)
        layers[i].feedForward(layers[i-1]);
};

std::vector<double> Net::getOutputVals()
{
    return layers[layers.size()-1].getOutputVals();
};
