#pragma once
#include <torch/extension.h>
#include "utils/neighbors.h"
#include <iostream>


at::Tensor example(at::Tensor query,
		   at::Tensor support){
	// std::vector<PointXYZ> queries_stl = std::vector<PointXYZ>((PointXYZ*)query.data_ptr<float>());
	// std::vector<PointXYZ> supports_stl = std::vector<PointXYZ>((PointXYZ*)support.data_ptr<float>());

	auto data_vec = query.data_ptr<float>();
	std::vector<PointXYZ>example((PointXYZ*) data_vec, (PointXYZ*) data_vec+5);

	for(int i=0; i < 15; i++)
	std::cerr<<data_vec[i]<<std::endl;

	//std::cerr<<data_vec<<std::endl;
	std::vector<int> neighbors_indices;

	at::Tensor out = torch::zeros({3, 5});
	for(int i=0; i<5; i++){
		out[0][i] = example[i].x;
		out[1][i] = example[i].y;
		out[2][i] = example[i].z;
	}


	return out;



}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("example", example);
}
