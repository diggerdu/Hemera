#include <vector>
#include <cstdlib>
#include <cassert>
#include "activation_function.h"


typedef std::vector<unsigned> Topo;

typedef std::vector<double> Synapses;


class Neuron;
typedef std::vector<Neuron> Layer;

// ************class Neuron **********
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned Idx, double (*activation_function)(double sum));
    void setOutputVal(double val) { outputVal = val; }
    double getOutputVal(void) const { return outputVal; }
    void feedForward(const Layer& prevLayer);
	void Neuron::setWeights(const Synapses &weights);
private:
    double outputVal;
	double bias;
    Synapses outputWeights;
    unsigned selfIdx;
    double (*activation)(double sum);
};

Neuron::Neuron(unsigned numOutputs, unsigned Idx, double (*activation_function)(double sum))
{
    activation = activation_function;
    for (unsigned synapseIdx = 0; synapseIdx < numOutputs; ++ synapseIdx)
    {
        outputWeights.push_back(0);
    }
	bias = 0;
	selfIdx = Idx;
}

void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;
    // Sum the previous layer's outputs 
    // Include the bias node from the previous layer.
    for (unsigned neuronIdx = 0; neuronIdx < prevLayer.size(); ++ neuronIdx)
        sum += prevLayer[neuronIdx].getOutputVal() * prevLayer[neuronIdx].outputWeights[selfIdx];
	sum += bias;
    outputVal = Neuron::activation(sum);
}

void Neuron::setWeights(const Synapses &weights)
{
	assert(outputWeights.size() == weights.size());
	outputWeights = weights;
}
// **************class Net ***********
class Net
{
public:
    Net(const Topo &topology);
    void feedForward(const std::vector<double> &inputVals);
    void getResults(std::vector<double> &resultVals) const;
private:
    std::vector<Layer> layers;     //layers[layerNum][neuronNum]
};


Net::Net(const Topo &topology)
{
	assert(topology.size() > 1);
    unsigned numLayers = topology.size();
    for (unsigned layerIdx = 0; layerIdx < numLayers; ++layerIdx)
    {
        /* construct a new layer */
        layers.push_back(Layer());
        unsigned numOutputs = layerIdx == topology.size() - 1 ? 0 : topology[layerIdx + 1];
        /* add a bias neuron to layer */
        for (unsigned neuronIdx = 0; neuronIdx < topology[layerIdx]; ++ neuronIdx)
        {
            layers.back().push_back(Neuron(numOutputs, neuronIdx, sigmoid));
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == layers[0].size());
    // assign (latch) the input values into the inpu neurons
    for (unsigned neuronIdx = 0; neuronIdx < inputVals.size(); ++ neuronIdx)
    {
       layers[0][neuronIdx].setOutputVal(inputVals[neuronIdx]);
    }
    
    //Forward propagate
    for (unsigned layerIdx = 1; layerIdx < layers.size(); ++layerIdx)
    {
        Layer &prevLayer = layers[layerIdx - 1];
        for (unsigned neuronIdx = 0; neuronIdx < layers[layerIdx].size(); ++neuronIdx)
        {
            layers[layerIdx][neuronIdx].feedForward(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &resultVals) const
{
	Layer outputLayer = layers.back();
	for (unsigned neuronIdx = 0; neuronIdx < outputLayer.size(); ++neuronIdx)
	{
		resultVals.push_back(outputLayer[neuronIdx].getOutputVal());
	}
	assert(resultVals.size() == outputLayer.size());
}