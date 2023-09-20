

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include "movie.h"
#include "csv.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kmeans.h"
#define N 64
#define TPB 32
#define K 3
#define MAX_ITER 20

using namespace std;
using namespace CsvProc;
using namespace MovieData;
using namespace KmeansCluster;
const string trainFile = "train_moviedata2.csv";
const string testFile = "test_moviedata2.csv";

///////////////////////////////////////////////////Clustring/////////////////////////////////////////////////////////////////////
__device__ float distance(float x1, float x2)
{
	return sqrt((x2 - x1) * (x2 - x1));
}

__global__ void kMeansClusterAssignment(float* d_datapoints, int* d_clust_assn, float* d_centroids)
{	
	printf("kMeansClusterAssignment init.22\n");
	//get idx for this datapoint
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for (int c = 0; c < K; ++c)
	{
		float dist = distance(d_datapoints[idx], d_centroids[c]);

		if (dist < min_dist)
		{
			min_dist = dist;
			closest_centroid = c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx] = closest_centroid;
    printf("kMeansClusterAssignment end.\n");
}


__global__ void kMeansCentroidUpdate(float* d_datapoints, int* d_clust_assn, float* d_centroids, int* d_clust_sizes)
{
	printf("kMeansCentroidUpdate init.\n");
	//get idx of thread at grid level
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float s_datapoints[TPB];
	s_datapoints[s_idx] = d_datapoints[idx];

	__shared__ int s_clust_assn[TPB];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if (s_idx == 0)
	{
		float b_clust_datapoint_sums[K] = { 0 };
		int b_clust_sizes[K] = { 0 };

		for (int j = 0; j < blockDim.x; ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[clust_id] += s_datapoints[j];
			b_clust_sizes[clust_id] += 1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for (int z = 0; z < K; ++z)
		{
			atomicAdd(&d_centroids[z], b_clust_datapoint_sums[z]);
			atomicAdd(&d_clust_sizes[z], b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids
	if (idx < K) {
        const float epsilon = 1e-10;  // or another small value of your choice
        d_centroids[idx] = d_centroids[idx] / (d_clust_sizes[idx] + epsilon);
	}
    printf("kMeansCentroidUpdate end.\n");

}

///////////////////////////////////////////////////Clustring/////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////Testing/////////////////////////////////////////////////////////////////////
__device__ float distanceFromCentroid(float* a, float* b, int length) {
    float dist = 0.0f;
    for (int i = 0; i < length; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}
__device__ float predict(float* datapoint, float* centroids, int numAttributes) {
    int closestCluster = 0;
    float minDistance = INFINITY;

    // Find the closest centroid
    for (int c = 0; c < K; c++) {
        float* currentCentroid = &centroids[c * numAttributes];
        float dist = distanceFromCentroid(datapoint, currentCentroid, numAttributes);
        if (dist < minDistance) {
            minDistance = dist;
            closestCluster = c;
        }
    }

    // For simplicity, let's assume the last attribute of the centroid 
    // is the mean gross for movies in that cluster.
    // You would need to ensure that this is the case when you're calculating the centroids.
    return centroids[closestCluster * numAttributes + numAttributes - 1];
}

__global__ void testKernel(float* d_datapoints, float* d_centroids, float* d_expected_gross, float* d_errorPercent, int numMovies, int numAttributes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numMovies) return;

    // Get the starting address for the datapoints of the current movie
    float* movieData = d_datapoints + idx * numAttributes;

    // Predict the gross for the movie
    float predicted = predict(movieData, d_centroids, numAttributes);
    float expected = d_expected_gross[idx];
    float difference = fabsf(predicted - expected);

    // Compute the error percentage for this movie
    if (expected != 0) {
        d_errorPercent[idx] = difference / expected;
    }
    else {
        d_errorPercent[idx] = 0;  // or another suitable value when expected gross is 0
    }
}

///////////////////////////////////////////////////Testing/////////////////////////////////////////////////////////////////////

int main()
{
    chrono::high_resolution_clock::time_point time1 = chrono::high_resolution_clock::now();
    printf("load data init.\n");

    //open the train file
    ifstream trainF(trainFile);
    if (!trainF) {
        cerr << "Couldn't find train file" << endl;
        exit(-20);
    }

    //process the train csv
    Csv trainCsv(trainF, 24);
    auto train = trainCsv.getDataMatrix();
    vector<Movie> trainset ,testset;
    printf("load data Success.\n");
    // go through train matrix and make the movie objects
    for (int i = 0; i < train.size(); ++i) {
        Movie movie = Movie(train[i]);
        movie.normalize(trainCsv);
        trainset.push_back(movie);
    }
    //open the test file
    ifstream testF(testFile);
    //check if file path exits
    if (!testF) {
        cerr << "Couldn't find test file" << endl;
        exit(-20);
    }
    //process the test csv
    Csv testCsv(testF, 24);
    auto test = testCsv.getDataMatrix();

    //go through train matrix and make the movie objects
    for (int i = 0; i < test.size(); ++i) {
        Movie movie = Movie(test[i]);
        movie.normalize(testCsv);
        testset.push_back(movie);
    }
    // Assuming each Movie has the same number of attributes
    int numMovies = trainset.size();
    int numAttributes = trainset[0].getSize();

    float* h_datapoints = new float[numMovies * numAttributes];
    printf("h_datapoints Success.\n");
   
    for (int i = 0; i < numMovies; i++) {
        for (int j = 0; j < numAttributes; j++) {
            h_datapoints[i * numAttributes + j] = trainset[i][j];
        }
    }
    printf(" for (int i = 0; i < numMovies; i++) Success.\n");
   
    printf("kMeans init.\n");
    //allocate memory on the device for the data points
    float* d_datapoints = 0;
    //allocate memory on the device for the cluster assignments
    int* d_clust_assn = 0;
    //allocate memory on the device for the cluster centroids
    float* d_centroids = 0;
    //allocate memory on the device for the cluster sizes
    int* d_clust_sizes = 0;

    printf("allocate memory Success.\n");
   
    cudaMalloc(&d_datapoints, numMovies * numAttributes * sizeof(float));
    cudaMalloc(&d_clust_assn, numMovies * sizeof(int));
    cudaMalloc(&d_centroids, K * numAttributes * sizeof(float));
    cudaMalloc(&d_clust_sizes, K * sizeof(int));

    printf("cudaMalloc Success.\n");
   
    float* h_centroids = (float*)malloc(K * numAttributes * sizeof(float));
    int* h_clust_sizes = (int*)malloc(K * sizeof(int));
   
    srand(time(0));
    printf("srand Success.\n");
    
    //initialize centroids
    for (int c = 0; c < K; ++c)
    {
        for (int d = 0; d < numAttributes; ++d) {
            h_centroids[c * numAttributes + d] = (float)rand() / (double)RAND_MAX;
            printf("%f ", h_centroids[c * numAttributes + d]);
        }
        printf("\n");
        h_clust_sizes[c] = 0;
    }
    printf("initialize centroids Success.\n");
    
    cudaMemcpy(d_centroids, h_centroids, K * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datapoints, h_datapoints, numMovies * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clust_sizes, h_clust_sizes, K * sizeof(int), cudaMemcpyHostToDevice);
    printf("cudaMemcpy Success.\n");
   
    int cur_iter = 1;

    while (cur_iter < MAX_ITER)
    {
        printf("MAX_ITER inside2.\n");
        cudaError_t err;
        err = cudaMemcpy(d_datapoints, h_datapoints, numMovies * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying to device: %s\n", cudaGetErrorString(err));
        }
        //call cluster assignment kernel
        
        kMeansClusterAssignment << <(numMovies + TPB - 1) / TPB, TPB >> > (d_datapoints, d_clust_assn, d_centroids);
        
        printf("kMeansClusterAssignment success.\n");
        
        //copy new centroids back to host 
        cudaMemcpy(h_centroids, d_centroids, K * numAttributes * sizeof(float), cudaMemcpyDeviceToHost);
        printf("copy new centroids back to host success.\n");
        
        for (int i = 0; i < K; ++i) {
            for (int d = 0; d < numAttributes; ++d) {
                printf("Iteration %d: centroid %d, dim %d: %f\n", cur_iter, i, d, h_centroids[i * numAttributes + d]);
            }
        }

        printf(" for (int i = 0; i < K; ++i) success.\n");
        
        //reset centroids and cluster sizes (will be updated in the next kernel)
        cudaMemset(d_centroids, 0.0, K * numAttributes * sizeof(float));
        cudaMemset(d_clust_sizes, 0, K * sizeof(int));
        printf(" cudaMemset success1.\n");
      
        //call centroid update kernel
        kMeansCentroidUpdate << <(numMovies + TPB - 1) / TPB, TPB >> > (d_datapoints, d_clust_assn, d_centroids, d_clust_sizes);
        printf(" kMeansCentroidUpdate success1.\n");
        cur_iter += 1;
    }
    
    cudaFree(d_datapoints);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids);
    cudaFree(d_clust_sizes);

    free(h_centroids);
    free(h_clust_sizes);
    delete[] h_datapoints;
    printf(" clustering end 69\n");

    /////////////////////////////////////////
    // //loop through and predict the gross
    printf(" start testing\n");
    float* h_expected_gross = new float[numMovies];
    for (int i = 0; i < numMovies; i++) {
        h_expected_gross[i] = testset[i][GROSS]; // assuming GROSS is a constant index of the gross feature
    }
    float* d_expected_gross = nullptr;
    cudaMalloc(&d_expected_gross, numMovies * sizeof(float));
    cudaMemcpy(d_expected_gross, h_expected_gross, numMovies * sizeof(float), cudaMemcpyHostToDevice);

    float* d_errorPercent = nullptr;
    cudaMalloc(&d_errorPercent, numMovies * sizeof(float));
    cudaMemset(d_errorPercent, 0, numMovies * sizeof(float));
    printf(" float* d_errorPercent = nullptr;\n");
    
    testKernel << <(numMovies + TPB - 1) / TPB, TPB >> > (d_datapoints, d_centroids, d_expected_gross, d_errorPercent, numMovies, numAttributes);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    printf(" testKernel success\n");
    
    float* h_errorPercent = new float[numMovies];
    cudaMemcpy(h_errorPercent, d_errorPercent, numMovies * sizeof(float), cudaMemcpyDeviceToHost);
    printf(" cudaMemcpy h_errorPercent,d_errorPercent,success\n");
    
    for (int i = 0; i < numMovies; i++) {
        printf("Error percent for movie %d: %f\n", i, h_errorPercent[i]);
    }

    float averageError = 0.0f;
    for (int i = 0; i < numMovies; i++) {
        averageError += h_errorPercent[i];
    }
    printf(" float averageError,success\n");
    
    averageError /= numMovies;

    cout << "Average Error: " << averageError * 100 << "%" << endl;
    
    cudaFree(d_expected_gross);
    cudaFree(d_errorPercent);
    delete[] h_expected_gross;
    delete[] h_errorPercent;
    
    ////////////////////////////////////////


    chrono::high_resolution_clock::time_point time2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(time2 - time1).count();

    cout << "========================================" << endl;
    cout << "Approximate Runtime : " << duration << " milliseconds" << endl;
    cout << "========================================" << endl;
    return 0;
}
