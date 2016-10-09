#include <vector>
#include <cstdlib>
#include <cassert>
#include "activation_function.h"
typedef std::vector<unsigned> Topo;
// ***********class Synapse *******
struct Synapse
{
    double weight;
    double deltaWeight;
};


typedef std::vector<Synapse> Synapses;


class Neuron;
typedef std::vector<Neuron> Layer;

// ************class Neuron **********
class Neuron
{
public:
    Neuron(unsigned numOutputs, double (*activation_function)(double sum));
    void setOutputVal(double val) { outputVal = val; }
    double getOutputVal(void) const { return outputVal; }
    void feedForward(const Layer& prevLayer);
private:
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double outputVal;
    Synapses outputWeights;
    unsigned selfIdx;
    double (*activation)(double sum);
    
};

Neuron::Neuron(unsigned numOutputs, double (*activation_function)(double sum))
{
    activation = activation_function;
    for (unsigned synapseIdx = 0; synapseIdx < numOutputs; ++ synapseIdx)
    {
        outputWeights.push_back(Synapse());
        outputWeights.back().weight = randomWeight();
    }
}

void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;
    // Sum the previous layer's outputs 
    // Include the bias node from the previous layer.
    for (unsigned neuronIdx = 0; neuronIdx < prevLayer.size(); ++ neuronIdx)
        sum + prevLayer[neuronIdx].getOutputVal() * prevLayer[neuronIdx].outputWeights[selfIdx].weight;
    outputVal = Neuron::activation(sum);
}

// **************class Net ***********
class Net
{
public:
    Net(const Topo &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targeVals);
    void getResults(std::vector<double> &resultVals) const;
private:
    std::vector<Layer> layers;     //layers[layerNum][neuronNum]
};


Net::Net(const Topo &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerIdx = 0; layerIdx < numLayers; ++layerIdx)
    {
        /* construct a new layer */
        layers.push_back(Layer());
        unsigned numOutputs = layerIdx == topology.size() - 1 ? 0 : topology[layerIdx + 1];
        /* add a bias neuron to layer */
        for (unsigned neuronIdx = 0; neuronIdx <= topology[layerIdx]; ++ neuronIdx)
        {
            layers.back().push_back(Neuron(numOutputs, sigmoid));
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == layers[0].size() - 1);
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