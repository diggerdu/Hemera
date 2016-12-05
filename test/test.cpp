#include <iostream>
#include <vector>
#include <algorithm>


int main()
{
		double test_double = 1.000000000000000000e+00;
	       	std::cout << test_double << std::endl;	
		int test[10] = {, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, };
		int step = 3;
		int *head;
		head = test;
		std::vector<int> tmp(3);
		for (int i = 0; i < 3; i++)
		{
			copy(head, head + step, tmp.begin());
			head = head + step + 1;
		};
		std::cout<<tmp[0]<<tmp[1]<<tmp[2]<<std::endl;
}

