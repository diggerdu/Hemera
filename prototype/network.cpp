#include "network_pre.h"

Node::Node(double(*activation_function)(double sum)):activate(activation_function)
{
    OutputVal = 0;
};

void Node::feedForward(const std::vector<double>& InputVals)
{
    double sum = 0;
    assert(InputVals.size() == Synapse.size());
    for (unsigned i = 0; i < Synapse.size(); i++)
        sum += InputVals[i]*Synapse[i];
    setOutputVal(activate(sum));
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