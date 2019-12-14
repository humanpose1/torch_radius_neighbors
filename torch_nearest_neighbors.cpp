#pragma once
#include <torch/extension.h>
#include "utils/neighbors.h"
#include <iostream>


at::Tensor radius_search(at::Tensor query,
			 at::Tensor support,
			 float radius, int max_num=-1, int mode=0){

	auto data_q = query.data_ptr<float>();
	auto data_s = support.data_ptr<float>();
	std::vector<PointXYZ> queries_stl = std::vector<PointXYZ>((PointXYZ*)data_q,
								  (PointXYZ*)data_q + query.size(0));
	std::vector<PointXYZ> supports_stl = std::vector<PointXYZ>((PointXYZ*)data_s,
								   (PointXYZ*)data_s + support.size(0));

	std::vector<long> neighbors_indices;
	int max_count = nanoflann_neighbors(queries_stl, supports_stl ,neighbors_indices, radius, max_num, mode);

	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	long* neighbors_indices_ptr = neighbors_indices.data();
	at::Tensor out;
	if(mode == 0)
		out = torch::from_blob(neighbors_indices_ptr, {queries_stl.size(), max_count}, options=options);
	else if(mode ==1)
		out = torch::from_blob(neighbors_indices_ptr, {neighbors_indices.size()/2, 2}, options=options);

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
	      " - mode : int number that indicate which format for the neighborhood"
	      " mode=0 mean a matrix of neighbors"
	      "mode=1 means a matrix of edges of size Num_edge x 2"
	      "return a tensor of size N1 x M where M is either max_num or the maximum number of neighbors found if mode = 0, if mode=1 return a tensor of size Num_edge x 2.",
	      "query"_a, "support"_a, "radius"_a, "max_num"_a=-1, "mode"_a=0);
}
