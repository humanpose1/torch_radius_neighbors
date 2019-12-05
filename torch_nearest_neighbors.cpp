#pragma once
#include <torch/extension.h>
#include "utils/neighbors.h"
#include <iostream>


at::Tensor radius_search(at::Tensor query,
			 at::Tensor support,
			 float radius, int max_num=-1){

	auto data_q = query.data_ptr<float>();
	auto data_s = support.data_ptr<float>();
	std::vector<PointXYZ> queries_stl = std::vector<PointXYZ>((PointXYZ*)data_q,
								  (PointXYZ*)data_q + query.size(0));
	std::vector<PointXYZ> supports_stl = std::vector<PointXYZ>((PointXYZ*)data_s,
								   (PointXYZ*)data_s + support.size(0));

	std::vector<long> neighbors_indices;
	int max_count = nanoflann_neighbors(queries_stl, supports_stl ,neighbors_indices, radius, max_num);
	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);


	long* neighbors_indices_ptr = neighbors_indices.data();
	at::Tensor out = torch::from_blob(neighbors_indices_ptr, {queries_stl.size(), max_count}, options=options);

	// for(int i=0; i < 10; i++){
	// 	std::cerr << neighbors_indices[i] << std::endl;
	// 	std::cerr << out[i] << std::endl;
	// }
	// int max_count = 17;

	// at::Tensor out = torch::zeros({queries_stl.size(), max_count}, at::kInt);

	// for(int i=0; i < out.size(0); i++){
	// 	for(int j=0; j < out.size(1); j++){
	// 		out[i][j] = neighbors_indices[max_count*i + j];
	//  		//out[i][j] = 0;
	// 	}
	// }

	return out.clone();
}
using namespace pybind11::literals;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("radius_search",
	      &radius_search,
	      "compute the radius search of a point cloud using nanoflann"
	      "-query : a pytorch tensor of size N1 x 3,. used to query the nearest neighbors"
	      "- support : a pytorch tensor of size N2 x 3. used to build the tree"
	      "-  radius : float number, size of the ball for the radius search."
	      "- max_num : int number, indicate the maximum of neaghbors allowed(if -1 then all the possible neighbors will be computed). "
	      "return a tensor of size N1 x M where M is either max_num or the maximum number of neighbors found",
	      "query"_a, "support"_a, "radius"_a, "max_num"_a=-1);
}
