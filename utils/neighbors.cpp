
// Taken from https://github.com/HuguesTHOMAS/KPConv

#include "neighbors.h"

template<typename scalar_t>
int nanoflann_neighbors(vector<scalar_t>& queries,
			vector<scalar_t>& supports,
			vector<long>& neighbors_indices,
			vector<float>& dists,
			float radius,
			int max_num,
			int mode){

	// Initiate variables
	// ******************

	// square radius

	const float search_radius = static_cast<float>(radius*radius);

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 1;


	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud<scalar_t> pcd;
	pcd.set(supports);
	//Cloud query
	PointCloud<scalar_t> pcd_query;
	pcd_query.set(queries);

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(15 /* max leaf */);

	// KDTree type definition
	typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<scalar_t, PointCloud<scalar_t> > ,
						     PointCloud<scalar_t>,
						     3 > my_kd_tree_t;

	// Pointer to trees
	my_kd_tree_t* index;
	index = new my_kd_tree_t(3, pcd, tree_params);
	index->buildIndex();
	// Search neigbors indices
	// ***********************

	// Search params
	nanoflann::SearchParams search_params;
	search_params.sorted = true;
	std::vector< std::vector<std::pair<size_t, scalar_t> > > list_matches(pcd_query.pts.size());

	for (auto& p0 : pcd_query.pts){

		// Find neighbors
		scalar_t query_pt[3] = { p0.x, p0.y, p0.z};
		list_matches[i0].reserve(max_count);
		std::vector<std::pair<size_t, scalar_t> >   ret_matches;


		const size_t nMatches = index->radiusSearch(&query_pt[0], search_radius, ret_matches, search_params);
		list_matches[i0] = ret_matches;
		if((size_t)max_count < nMatches) max_count = nMatches;
		i0++;


	}
	// Reserve the memory
	if(max_num > 0) {
		max_count = max_num;
	}
	if(mode == 0){

		neighbors_indices.resize(list_matches.size() * max_count);
		dists.resize(list_matches.size() * max_count);

		i0 = 0;

		for (auto& inds : list_matches){
			for (int j = 0; j < max_count; j++){
				if (j < inds.size()){
					neighbors_indices[i0 * max_count + j] = inds[j].first;
					dists[i0 * max_count + j] = (float) inds[j].second;
				}

				else {
					neighbors_indices[i0 * max_count + j] = -1;
					dists[i0 * max_count + j] = radius * radius;
				}
			}
			i0++;
		}

	}
	else if(mode == 1){
		int size = 0; // total number of edges
		for (auto& inds : list_matches){
			if((int)inds.size() <= max_count)
				size += inds.size();
			else
				size += max_count;
		}
		neighbors_indices.resize(size*2);
		dists.resize(size);
		int i0 = 0; // index of the query points
		int u = 0; // curent index of the neighbors_indices
		for (auto& inds : list_matches){
			for (int j = 0; j < max_count; j++){
				if(j < inds.size()){
					neighbors_indices[u] = inds[j].first;
					neighbors_indices[u + 1] = i0;
					dists[u/2] = (float) inds[j].second;
					u += 2;
				}
			}
			i0++;
		}


	}
	return max_count;




}
template<typename scalar_t>
int batch_nanoflann_neighbors (vector<scalar_t>& queries,
                               vector<scalar_t>& supports,
                               vector<long>& q_batches,
                               vector<long>& s_batches,
                               vector<long>& neighbors_indices,
			       vector<float>& dists,
                               float radius, int max_num,
			       int mode){


// Initiate variables
// ******************
// indices
	int i0 = 0;

// Square radius
	float r2 = radius * radius;

	// Counting vector
	int max_count = 0;


	// batch index
	long b = 0;
	long sum_qb = 0;
	long sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud<scalar_t> current_cloud;
	PointCloud<scalar_t> query_pcd;
	query_pcd.set(queries);
	vector<vector<pair<size_t, scalar_t> > > all_inds_dists(query_pcd.pts.size());

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
	typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<scalar_t, PointCloud<scalar_t> > , PointCloud<scalar_t> ,3 > my_kd_tree_t;

// Pointer to trees
	my_kd_tree_t* index;
    // Build KDTree for the first batch element
	current_cloud.set_batch(supports, sum_sb, s_batches[b]);
	index = new my_kd_tree_t(3, current_cloud, tree_params);
	index->buildIndex();
// Search neigbors indices
// ***********************
// Search params
	nanoflann::SearchParams search_params;
	search_params.sorted = true;
	for (auto& p0 : query_pcd.pts){
// Check if we changed batch

		if (i0 == sum_qb + q_batches[b]){
			sum_qb += q_batches[b];
			sum_sb += s_batches[b];
			b++;

// Change the points
			current_cloud.pts.clear();
			current_cloud.set_batch(supports, sum_sb, s_batches[b]);
// Build KDTree of the current element of the batch
			delete index;
			index = new my_kd_tree_t(3, current_cloud, tree_params);
			index->buildIndex();
		}
// Initial guess of neighbors size
		all_inds_dists[i0].reserve(max_count);
// Find neighbors
		scalar_t query_pt[3] = { p0.x, p0.y, p0.z};
		size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);
// Update max count

		if (nMatches > (size_t)max_count)
			max_count = nMatches;
// Increment query idx
		i0++;
	}
	// how many neighbors do we keep
	if(max_num > 0) {
		max_count = max_num;
	}
// Reserve the memory
	if(mode == 0){
		neighbors_indices.resize(query_pcd.pts.size() * max_count);
		dists.resize(query_pcd.pts.size() * max_count);
		i0 = 0;
		sum_sb = 0;
		sum_qb = 0;
		b = 0;

		for (auto& inds_dists : all_inds_dists){// Check if we changed batch

			if (i0 == sum_qb + q_batches[b]){
				sum_qb += q_batches[b];
				sum_sb += s_batches[b];
				b++;
			}

			for (int j = 0; j < max_count; j++){
				if (j < inds_dists.size()){
					neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
					dists[i0 * max_count + j] = (float) inds_dists[j].second;
				}
				else {
					neighbors_indices[i0 * max_count + j] = supports.size();
					dists[i0 * max_count + j] = radius * radius;
				}

			}

			i0++;
		}
		delete index;
	}
	else if(mode == 1){
		int size = 0; // total number of edges
		for (auto& inds_dists : all_inds_dists){
			if((int)inds_dists.size() <= max_count)
				size += inds_dists.size();
			else
				size += max_count;
		}
		neighbors_indices.resize(size * 2);
		dists.resize(size);
		i0 = 0;
		sum_sb = 0;
		sum_qb = 0;
		b = 0;
		int u = 0;
		for (auto& inds_dists : all_inds_dists){
			if (i0 == sum_qb + q_batches[b]){
				sum_qb += q_batches[b];
				sum_sb += s_batches[b];
				b++;
			}
			for (int j = 0; j < max_count; j++){
				if (j < inds_dists.size()){
					neighbors_indices[u] = inds_dists[j].first + sum_sb;
					neighbors_indices[u + 1] = i0;
					dists[u/2] = (float) inds_dists[j].second;
					u += 2;
				}
			}
			i0++;
		}
	}
	return max_count;
}
