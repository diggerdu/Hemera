#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

typedef std::vector<unsigned> Topo;




class Node;
class Layer;
class Net;


class LayerInfo
{
public:
    LayerInfo(double(*activation_function)(double sum), size_t NumNode);
    unsigned numnodes;
    double (*activate)(double sum);
};


class Node
{
public:
    Node(const std::vector<double>& Synapse, double(*activation_function)(double sum));
    void feedForward(const std::vector<double>& InputVals);
    void setOutputVal(const double Val) {OutputVal = Val;};
    double getOutputVal() {return OutputVal;};
private:
    double OutputVal;
    std::vector<double> synapse;
    double (*activate)(double sum);
};


class Layer
{
public:
    Layer(const std::vector<std::vector<double> >& Synapses, size_t NumNode, double(*activation_function)(double sum));
    void feedForward(Layer& Prevlayer);
    void setOutputVals(std::vector<double> inputVals);
    std::vector<double> getOutputVals();
private:
    std::vector<Node> nodes;
};



class Net
{
public:
    Net(const std::vector<LayerInfo>& Topo, const std::vector<std::vector<std::vector<double> > >& Synapses);
    void setInputVals(const std::vector<double> & InputVals);
    void feedForward();
    std::vector<double> getOutputVals();
private:
    std::vector<Layer> layers;
};

/*range from 0 to 1
 */
double sigmoid(double x)
{
    return 1/(1+exp(-1*x));
}


/*range from -1 to 1
 */
double sigmoid_h(double x)
{
    return 2/(1+exp(-x))-1;
}

/* range from -2 to 2
 */
double sigmoid_g(double x)
{
    return 4/(1+exp(-x))-2;
}

double relu(double x)
{
	return x > 0 ? x : 0;
}