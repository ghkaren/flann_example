#include <iostream>
#include "nanoflann/nanoflann.hpp"

#include "nanoflann/utils.h"
#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace nanoflann;

template <typename num_t>
void kdtree_demo(const size_t N)
{
  PointCloud<num_t> cloud;

  // construct a kd-tree index:
  typedef KDTreeSingleIndexDynamicAdaptor<
  L2_Simple_Adaptor<num_t, PointCloud<num_t> > ,
      PointCloud<num_t>,
      3 /* dim */
      > my_kd_tree_t;

  dump_mem_usage();

  my_kd_tree_t   index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );

  // Generate points:
  generateRandomPointCloud(cloud, N);

  num_t query_pt[3] = { 0.5, 0.5, 0.5 };

  // add points in chunks at a time
  size_t chunk_size = 100;
  for(size_t i = 0; i < N; i = i + chunk_size)
  {
    size_t end = min(size_t(i + chunk_size), N - 1);
    // Inserts all points from [i, end]
    index.addPoints(i, end);
  }

  // remove a point
  size_t removePointIndex = N - 1;
  index.removePoint(removePointIndex);

  dump_mem_usage();
  {
    cout << "Searching for 1 element..." << endl;
    // do a knn search
    const size_t num_results = 1;
    size_t ret_index;
    num_t out_dist_sqr;
    nanoflann::KNNResultSet<num_t> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr );
    index.findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    cout << "knnSearch(nn="<<num_results<<"): \n";
    cout << "ret_index=" << ret_index << " out_dist_sqr=" << out_dist_sqr << endl;
    cout << "point: (" << "point: (" << cloud.pts[ret_index].x
         << ", " << cloud.pts[ret_index].y
         << ", " << cloud.pts[ret_index].z << ")" << endl;
    cout << std::endl;
  }
  {
    // do a knn search searching for more than one result
    const size_t num_results = 5;
    cout << "Searching for " << num_results << " elements" << endl;
    size_t ret_index[num_results];
    num_t out_dist_sqr[num_results];
    nanoflann::KNNResultSet<num_t> resultSet(num_results);
    resultSet.init(ret_index, out_dist_sqr );
    index.findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    cout << "knnSearch(nn="<<num_results<<"): \n";
    cout << "Results: " << endl;
    for (size_t i = 0; i < resultSet.size(); ++i) {
      cout << "#" << i << ",\t"
           << "index: " << ret_index[i] << ",\t"
           << "dist: " << out_dist_sqr[i] << ",\t"
           << "point: (" << cloud.pts[ret_index[i]].x
           << ", " << cloud.pts[ret_index[i]].y
           << ", " << cloud.pts[ret_index[i]].z << ")" << endl;
    }
    cout << endl;
  }
  {
    // Unsorted radius search:
    std::cout << "Unsorted radius search" << std::endl;
    const num_t radius = 1;
    std::vector<std::pair<size_t, num_t> > indices_dists;
    RadiusResultSet<num_t, size_t> resultSet(radius, indices_dists);

    index.findNeighbors(resultSet, query_pt, nanoflann::SearchParams());

    // Get worst (furthest) point, without sorting:
    std::pair<size_t,num_t> worst_pair = resultSet.worst_item();
    cout << "Worst pair: idx=" << worst_pair.first << " dist=" << worst_pair.second << endl;
    cout << "point: (" << cloud.pts[worst_pair.first].x
         << ", " << cloud.pts[worst_pair.first].y
         << ", " << cloud.pts[worst_pair.first].z << ")" << endl;
    cout << endl;
  }
}

int main()
{
  // Randomize Seed
  srand(static_cast<unsigned int>(time(nullptr)));
  kdtree_demo<float>(1000000);
  kdtree_demo<double>(1000000);
  return 0;
}
