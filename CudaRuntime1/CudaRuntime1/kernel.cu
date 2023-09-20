
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>

// D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/CUDA/CUDA/
// D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_assignment/CUDA/CUDA/
// schools
//C:/Users/TARUMT/Desktop/DSPC_assignment/CUDA/CUDA/
// C:/Users/TARUMT/Desktop/CudaRuntime1/CudaRuntime1/
#include "C:/Users/TARUMT/Desktop/CudaRuntime1/CudaRuntime1/movie.h"
#include "C:/Users/TARUMT/Desktop/CudaRuntime1/CudaRuntime1/csv.h"
#include "kmeans.h"

using namespace std;
using namespace CsvProc;
using namespace MovieData;
const string trainFile = "C:/Users/TARUMT/Desktop/CudaRuntime1/CudaRuntime1/train_moviedata2.csv";
const string testFile = "C:/Users/TARUMT/Desktop/CudaRuntime1/CudaRuntime1/test_moviedata2.csv";



__global__ void addToClosestKernel(MovieData::Movie* movies, int n, MovieData::Movie* centroids, int* clusterIndex)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    MovieData::Movie amovie = movies[idx];


    /*float distance1 = distanceFromCluster(amovie, centroids[0]);
    float distance2 = distanceFromCluster(amovie, centroids[1]);
    float distance3 = distanceFromCluster(amovie, centroids[2]);


    float smallest = distance1;
    int cluster_id = 0;
    if (smallest > distance2) {
        smallest = distance2;
        cluster_id = 1;
    }
    if (smallest > distance3) {
        smallest = distance3;
        cluster_id = 2;
    }

    clusterIndex[idx] = cluster_id;*/
}

/*
__device__ float distanceFromCluster(const MovieData::Movie& amovie, const MovieData::Movie& centroid)
{
    float distance = 0;
    int numAttributes = ATTR_SIZE; // ATTR_SIZE seems to be the constant for the number of attributes

    for (int i = 0; i < numAttributes; i++) {
        float diff = amovie[i] - centroid[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}
*/
int main()
{
    cout << "Program started.\n";
    cout << "Starting K-Means clustering...\n";

    cout << "load CSV.\n";
    vector<vector<float>> train, test;
    vector<Movie> trainset, testset;

    //open the train file
    ifstream trainF(trainFile);
    if (!trainF) {
        cerr << "Couldn't find train file" << endl;
        exit(-20);
    }

    //open the test file
    ifstream testF(testFile);
    if (!testF) {
        cerr << "Couldn't find test file" << endl;
        exit(-20);
    }

    //process the train csv
    Csv trainCsv(trainF, 24);
    train = trainCsv.getDataMatrix();

    for (int i = 0; i < train.size(); ++i) {
        float dataArray[ATTR_SIZE] = { 0.0f };
        int numElementsToCopy = std::min(ATTR_SIZE, static_cast<int>(train[i].size()));
        std::copy(train[i].begin(), train[i].begin() + numElementsToCopy, dataArray);
        Movie movie(dataArray);
        movie.normalize(trainCsv);
        trainset.push_back(movie);
    }

    //process the test csv
    Csv testCsv(testF, 24);
    test = testCsv.getDataMatrix();

    for (int i = 0; i < test.size(); ++i) {
        float dataArray[ATTR_SIZE] = { 0.0f };
        int numElementsToCopy = std::min(ATTR_SIZE, static_cast<int>(test[i].size()));
        std::copy(test[i].begin(), test[i].begin() + numElementsToCopy, dataArray);
        Movie movie(dataArray);
        movie.normalize(testCsv);
        testset.push_back(movie);
    }



    cout << "load CSV done.\n";
    cout << "start CUDA done.\n";

    MovieData::Movie* d_movies, * d_centroids;
    int* d_clusterIndex;

    // Allocate memory on GPU
    cudaMalloc(&d_movies, trainset.size() * sizeof(MovieData::Movie));
    cudaMalloc(&d_centroids, 3 * sizeof(MovieData::Movie));  // Assuming 3 centroids
    cudaMalloc(&d_clusterIndex, trainset.size() * sizeof(int));

    // Copy data to GPU
    std::vector<MovieData::Movie> current = { trainset[0], trainset[1], trainset[2] };
    cudaMemcpy(d_movies, trainset.data(), trainset.size() * sizeof(MovieData::Movie), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, current.data(), 3 * sizeof(MovieData::Movie), cudaMemcpyHostToDevice);

    //// Define block and grid sizes
    int blockSize = 256;
    int gridSize = (trainset.size() + blockSize - 1) / blockSize;
    cout << "CUDA .\n";

    //// Launch the kernel
    //addToClosestKernel << <gridSize, blockSize >> > (d_movies, trainset.size(), d_centroids, d_clusterIndex);

    //// Copy the cluster indices back to host memory
    int* clusterIndices = new int[trainset.size()];
    cudaMemcpy(clusterIndices, d_clusterIndex, trainset.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_movies);
    cudaFree(d_centroids);
    cudaFree(d_clusterIndex);
    cout << "Ending cuda 111.\n";
    return 0;
}

