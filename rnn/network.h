#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>


class Node;
class Layer;
class Net;


class LayerInfo
{
public:
    LayerInfo(double(*activation_function)(double sum), size_t NumNode);
    size_t numnodes;
    double (*activate)(double sum);
};


class Node
{
public:
    Node(const std::vector<double>& Synapse, const double m_bias, double(*activation_function)(double sum));
    void feedForward(const std::vector<double>& InputVals);
    void setOutputVal(const double Val) {OutputVal = Val;};
    void init();
    double getOutputVal() {return OutputVal;};
private:
    double OutputVal;
    std::vector<double> synapse;
    double bias;
    double (*activate)(double sum);
};


class Layer
{
public:
    Layer(const std::vector<std::vector<double> >& Synapses, const std::vector<double>& Biases, size_t NumNode, double(*activation_function)(double sum));
    void init();
    void feedForward(Layer& Prevlayer);
    void setOutputVals(std::vector<double> inputVals);
    std::vector<double> getOutputVals();
private:
    std::vector<Node> nodes;
};



class Net
{
public:
    Net(const std::vector<LayerInfo>& Topo, const std::vector<std::vector<std::vector<double> > >& Synapses, const std::vector<std::vector<double> > Biases);
    void setInputVals(const std::vector<double> & InputVals);
    void feedForward();
    void init();
    std::vector<double> getOutputVals();
private:
    std::vector<Layer> layers;
};

/*range from 0 to 1
 */
extern double sigmoid(double x);


/*range from -1 to 1
 */
extern double sigmoid_h(double x);

/* range from -2 to 2
 */
extern double sigmoid_g(double x);

extern double relu(double x);

extern double tanh(double x);
