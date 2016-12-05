#include "ulti.h"
//#include <iostream>

std::vector<std::vector<std::vector<double> > > arr2vec3d(const double *arr, const size_t *size, const size_t depth)
{
	const double *head;
	head = arr;
	std::vector<std::vector<std::vector<double> > > vec3d;
	std::vector<std::vector<double> > ZeroPad;
	for (size_t i = 0; i < size[0]; i++)
	{
		std::vector<double> tmp;
		tmp.push_back(0);
		ZeroPad.push_back(tmp);
	}
	vec3d.push_back(ZeroPad);

	
	for (size_t i = 1; i < depth; i++)
	{
		std::vector<std::vector<double > > vec2d;
		size_t weights_len = size[i-1] + size[i];
		for (size_t j = 0; j < size[i]; j++)
		{
			std::vector<double> tmp(weights_len);
			copy(head, head + weights_len, tmp.begin());
			vec2d.push_back(tmp);
			head = head + weights_len;	
		}
		vec3d.push_back(vec2d);
	}

	return vec3d;
};

std::vector<std::vector<double> > arr2vec2d(const double *arr, const size_t *size, const size_t depth)
{
	//std::cout<<arr[0]<<std::endl;
	const double *head;
	head = arr;
	std::vector<std::vector<double> > vec2d;
	vec2d.push_back(std::vector<double>(size[0], 0));
	for (size_t i = 1; i < depth; i++)
	{
		std::vector<double> tmp(size[i]);
		copy(head, head + size[i], tmp.begin());
		vec2d.push_back(tmp);
		head = head + size[i];
	}
	return vec2d;
};
