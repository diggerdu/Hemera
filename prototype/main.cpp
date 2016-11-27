#include<iostream>
#include<vector>
#include"network.h"
int main()
{
    std::vector<double> inputvals;
    Topo topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net sampleNet(topology);
    std::vector<double> sampleInput;
    sampleInput.push_back(1);
    sampleInput.push_back(1);
    sampleInput.push_back(1);
    sampleNet.feedForward(sampleInput);

    std::vector<double> sampleOutput;
    sampleNet.getResults(sampleOutput);
    std::cout<<sampleOutput.back()<<std::endl;



    double nobody;
    std::cin >> nobody;
    return 0;
}
