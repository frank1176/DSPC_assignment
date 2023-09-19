
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
#include "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_assignment/CUDA/CUDA/movie.h"
#include "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_assignment/CUDA/CUDA/csv.h"
#include "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_assignment/CUDA/CUDA/kmeans.h"

using namespace std; 
using namespace CsvProc;
using namespace MovieData;
const string trainFile = "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_assignment/CUDA/CUDA/train_moviedata2.csv";
const string testFile = "D:/TARC/Year3/sem7/Distributed Systems and Parallel Computing/DSPC_assignment/CUDA/CUDA/test_moviedata2.csv";



cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

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

    cout << "load CSV done1.\n";
    return 0;
}

