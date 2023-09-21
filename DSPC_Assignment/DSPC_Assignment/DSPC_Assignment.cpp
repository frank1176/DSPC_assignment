// DSPC_Assignment.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <thread>
#include "movie.h"
#include "csv.h"
#include "kmeans.h"
#include <mutex>

using namespace std;
using namespace CsvProc;
using namespace MovieData;
using namespace KmeansCluster;  


const string trainFile = "train_moviedata2.csv";
const string testFile = "test_moviedata2.csv";


void runAlgorithm(vector<Movie>& train, vector<Movie>& test, Csv& t1, Csv& t2) {
    cout << "Movie Gross Predicter: " << endl;
    cout << "========================================" << endl;
    cout << "Enter 1 to run K-Means algorithm" << endl;
    cout << "Enter 2 to run K-Means algorithm(OpenMP)" << endl;
    cout << "Enter 3 to run K-Means algorithm(Thread)" << endl;
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

    case 2: {
        omp_set_num_threads(12);
        chrono::high_resolution_clock::time_point time1 = chrono::high_resolution_clock::now();

        // Create an instance of KMeansOpenMP
        KMeansOpenMP km2 = KMeansOpenMP();

        // Initialize the KMeansOpenMP algorithm with train data
        km2.initialize(train);

        // Run the KMeansOpenMP clustering algorithm
        km2.cluster();

        // Loop through and predict the gross for test data
        float errPercent = 0, count = 0;
        for (Movie& m : test) {
            float expected = km2.predict(m);
            float actual = m[GROSS];
            float difference = fabsf(expected - actual);
            errPercent += (difference / expected);
            ++count;
        }

        chrono::high_resolution_clock::time_point time2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(time2 - time1).count();
        cout << "========================================" << endl;
        cout << "K-Means OpenMP Average Error: " << (errPercent / count) * 100 << "%" << ", Approximate Runtime: " << duration << " milliseconds" << endl;
        cout << "========================================" << endl;
        break;
    }

    //option 3
    case 3:
    {

        cout << "========================================" << endl;
        cout << "Kmeans thread" << endl;
        cout << "========================================" << endl;
        chrono::high_resolution_clock::time_point time1 = chrono::high_resolution_clock::now();
        KmeansThread km3 = KmeansThread();
        // Initialize the KmeansPthread algorithm with train data
        km3.initialize(train);

        // Run the KmeansPthread clustering algorithm
        km3.cluster();
        
        float errPercent = 0, count = 0;
        for (Movie& m : test) {
            float expected = km3.predict(m);
            float actual = m[GROSS];
            float difference = fabsf(expected - actual);
            errPercent += (difference / expected);
            ++count;
        }
        chrono::high_resolution_clock::time_point time2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(time2 - time1).count();
        cout << "K-Means PThread Average Error: " << (errPercent / count) * 100 << "%" << ", Approximate Runtime: " << duration << " milliseconds" << endl;
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

