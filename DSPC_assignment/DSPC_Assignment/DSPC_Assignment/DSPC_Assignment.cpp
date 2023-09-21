// DSPC_Assignment.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>

#include "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_Assignment/DSPC_Assignment/movie.h"
#include "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_Assignment/DSPC_Assignment/csv.h"
#include "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_Assignment/DSPC_Assignment/kmeans.h"




#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace CsvProc;
using namespace MovieData;
using namespace KmeansCluster;  

//const string trainFile = "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC/Try/Try/train_moviedata.csv";
//const string testFile = "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC/Try/Try/test_moviedata.csv";
const string trainFile = "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_Assignment/DSPC_Assignment/train_moviedata2.csv";
const string testFile = "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_Assignment/DSPC_Assignment/test_moviedata2.csv";

//__global__ void assignClusters(float* data, float* centroids, int* assignments, int numPoints, int numCentroids, int dimensions) {
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//    if (tid < numPoints) {
//        float minDist = 1e10;  // A large value
//        int minIndex = -1;
//
//        for (int centroid = 0; centroid < numCentroids; centroid++) {
//            float dist = 0.0;
//
//            for (int dim = 0; dim < dimensions; dim++) {
//                float diff = data[tid * dimensions + dim] - centroids[centroid * dimensions + dim];
//                dist += diff * diff;
//            }
//
//            if (dist < minDist) {
//                minDist = dist;
//                minIndex = centroid;
//            }
//        }
//
//        assignments[tid] = minIndex;
//    }
//}

void runAlgorithm(vector<Movie>& train, vector<Movie>& test, Csv& t1, Csv& t2) {
    cout << "Movie Gross Predicter: " << endl;
    cout << "========================================" << endl;
    cout << "Enter 1 to run K-Means algorithm" << endl;
    cout << "Enter 2 to run K-Means algorithm(OpenMP)" << endl;
    cout << "Enter 3 to run K-Means algorithm(CUDA)" << endl;
    cout << "========================================" << endl;
    cout << "Press any numbers aside from 1,2,3 to exit..." << endl;
    cout << "========================================" << endl;
    cout << endl;
    int algo;
    cin >> algo;

    switch (algo) {
        //option 1
    case 1:
    {
        chrono::high_resolution_clock::time_point time1 = chrono::high_resolution_clock::now();
        //K-Means Algorithm
        KMeans km = KMeans();
        //Intialize the set
        km.initialize(train);
        //Make the cluster
        km.cluster();
        //loop through and predict the gross
        float errPercent2 = 0, count2 = 0;
        for (Movie& m : test) {
            float expected = km.predict(m);
            float actual = m[GROSS];
            float difference = fabsf(expected - actual);
            errPercent2 += (difference / expected);
            ++count2;
        }
        chrono::high_resolution_clock::time_point time2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(time2 - time1).count();
        cout << "========================================" << endl;
        cout << "K-Means Average Error: " << (errPercent2 / count2) * 100 << "%" << ", Approximate Runtime: " << duration << " milliseconds" << endl;
        cout << "========================================" << endl;
        break;
    }

    //option 2
    case 2:
    {
        cout << "========================================" << endl;
        cout << "Kmeans OpenMP" << endl;
        cout << "========================================" << endl;
    }

    //option 3
    case 3:
    {
        //cout << "========================================" << endl;
        //cout << "Kmeans CUDA" << endl;

        //float* d_data;  // Pointer for data on the GPU
        ////float* h_data;  // Assuming you define this elsewhere and fill it with the correct data.
        //int number_of_data_points = 10000; // Define this appropriately.
        //float* h_data = new float[number_of_data_points];

        //int data_size = sizeof(float) * number_of_data_points;
        //for (int i = 0; i < number_of_data_points; i++) {
        //    h_data[i] = static_cast<float>(i);
        //}
        //cudaError_t err = cudaMalloc((void**)&d_data, data_size);  // Allocate memory on GPU
        //if (err != cudaSuccess) {
        //    cerr << "Error in cudaMalloc: " << cudaGetErrorString(err) << endl;
        //    exit(-1);  // Exit the entire program
        //}

        //err = cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);  // Copy data from host to device
        //if (err != cudaSuccess) {
        //    cerr << "Error in cudaMemcpy (host to device): " << cudaGetErrorString(err) << endl;
        //    exit(-1);  // Exit the entire program
        //}

        //// ... computations ... (You should define and launch a CUDA kernel here)


        //cudaFree(d_data);  // Free GPU memory
        //delete[] h_data;
        //cout << "Kmeans CUDA END" << endl;
        //cout << "========================================" << endl;

        cout << "========================================" << endl;
        cout << "Kmeans CUDA" << endl;
        cout << "========================================" << endl;
        break;
    }
    default:
    {
        cout << "Exiting Program..." << endl;
        exit(-1);
        break;
    }
    }
}


int main(int argc, const char* argv[]) {
    // initialize matrices and vectors needed
    vector<vector<float>>train, test;
    vector<Movie>trainset, testset;

    //open the train file
    ifstream trainF(trainFile);
    //check if file path exits
    if (!trainF) {
        cerr << "Couldn't find train file" << endl;
        exit(-20);
    }

    //open the test file
    ifstream testF(testFile);
    //check if file path exits
    if (!testF) {
        cerr << "Couldn't find test file" << endl;
        exit(-20);
    }

    //process the train csv
    Csv trainCsv(trainF, 24);
    train = trainCsv.getDataMatrix();

    //go through train matrix and make the movie objects
    for (int i = 0; i < train.size(); ++i) {
        Movie movie = Movie(train[i]);
        movie.normalize(trainCsv);
        trainset.push_back(movie);
    }

    //process the test csv
    Csv testCsv(testF, 24);
    test = testCsv.getDataMatrix();

    //go through train matrix and make the movie objects
    for (int i = 0; i < test.size(); ++i) {
        Movie movie = Movie(test[i]);
        movie.normalize(testCsv);
        testset.push_back(movie);
    }

    while (true) {
        runAlgorithm(trainset, testset, trainCsv, testCsv);
    }


    return 0;
}

