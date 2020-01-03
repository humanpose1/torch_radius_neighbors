

#include "cloud.h"
#include "nanoflann.hpp"
#include <set>
#include <cstdint>

using namespace std;


template<typename scalar_t>
int nanoflann_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
			vector<long>& neighbors_indices,
			vector<float>& dists,
			float radius, int max_num, int mode);


template<typename scalar_t>
int batch_nanoflann_neighbors(vector<scalar_t>& queries,
			      vector<scalar_t>& supports,
			      vector<long>& q_batches,
			      vector<long>& s_batches,
			      vector<long>& neighbors_indices,
			      vector<float>& dists,
			      float radius, int max_num, int mode);
