//
//  KmeansPthread.cpp
//  MovieGross
//
//
//  Copyright � 2016 ArsenKevinMD. All rights reserved.
//

#include <stdio.h>
#include "movie.h"
#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>

using namespace std;
using namespace MovieData;

//K-Means Clustering Namespace
namespace KmeansCluster {
    std::mutex mtx;
    //find the closest cluster to add
    void KmeansPthread::addToClosest(MovieData::Movie& amovie) {
        //check if county passed in centroid
        if (current[0] == amovie || current[1] == amovie || current[2] == amovie) return;


        //get the distance from the centroid to the county
        float distance1 = distanceFromCluster(amovie, cluster1);
        float distance2 = distanceFromCluster(amovie, cluster2);
        float distance3 = distanceFromCluster(amovie, cluster3);

        //check which distance is the smallest
        float smallest = distance1;
        if (smallest > distance2) {
            smallest = distance2;
        }
        if (smallest > distance3) {
            smallest = distance3;
        }

        //based on the smallest distance add to cluster
        if (smallest == distance1) {
            cluster1.push_back(amovie);
        }
        else if (smallest == distance2) {
            cluster2.push_back(amovie);
        }
        else {
            cluster3.push_back(amovie);
        }
    }

    //method to predict movie gross
    float KmeansPthread::predict(MovieData::Movie aMovie) {
        //get the distance from the centroid to the county
        float distance1 = distanceFromCluster(aMovie, cluster1);
        float distance2 = distanceFromCluster(aMovie, cluster2);
        float distance3 = distanceFromCluster(aMovie, cluster3);

        //check which distance is the smallest
        float smallest = distance1;
        if (smallest > distance2) {
            smallest = distance2;
        }
        if (smallest > distance3) {
            smallest = distance3;
        }

        //based on the smallest distance add to cluster
        float total = 0, size = 0;

        if (smallest == distance1) {
            for (Movie m : cluster1) {
                total += m[GROSS];
                size += 1;
            }
        }
        else if (smallest == distance2) {
            for (Movie m : cluster2) {
                total += m[GROSS];
                size += 1;
            }
        }
        else {
            for (Movie m : cluster3) {
                total += m[GROSS];
                size += 1;
            }
        }
        return total / size;
    }

    //method to initialize
    void KmeansPthread::initialize(vector<Movie> movies) {
        //initialize random
        srand(time(NULL));

        //get three random indexes
        int index1 = rand() % 100;
        int index2 = rand() % 100 + 100;
        int index3 = rand() % 200 + 300;

        //get the three random centroids
        Movie first = movies[index1];
        Movie second = movies[index2];
        Movie third = movies[index3];
        //push them into the clusters and setup centroids
        cluster1.push_back(first);
        cluster2.push_back(second);
        cluster3.push_back(third);
        current.push_back(first);
        current.push_back(second);
        current.push_back(third);
        //go through counties and add to each cluster
        for (Movie c : movies) {
            //cout << "Movie c : movies" << endl;
            addToClosest(c);
            //cout << "addToClosest(c);" << endl;
            all.push_back(c);
            //cout << "all.push_back(c);" << endl;
        }
        //cout << "  for(Movie c : movies) done" << endl;
    }

    //method to get the mean of a cluster
    vector<float> KmeansPthread::mean(std::vector<Movie>& cluster) {
        //cout << "Inside mean() function" << endl;

        vector<float>totals;
        for (int i = 0; i < 25; ++i) {
            totals.push_back(0);
        }

        //cout << "Starting to tally the total sum..." << endl;
        //go through and tally the total sum
        int movieCount = 0; // To keep track of which movie we're processing
        for (Movie c : cluster) {
            //cout << "Processing movie " << movieCount << endl;

            for (int i = 0; i < 23; ++i) {
                if (i != GROSS) {
                    if (c.getSize() <= i) { // checking if the movie's data size is less than or equal to current index
                        //cout << "Warning: Movie " << movieCount << " does not have data at index " << i << "." << endl;
                        continue; // skip to the next iteration
                    }
                    //cout << "Accessing value at index " << i << ": " << c[i] << endl;
                    totals[i] += c[i];
                    if (i == 24) { // if this is the last index, just to check if we're reaching here
                        //cout << "Successfully accessed value at index 24 for movie " << movieCount << endl;
                    }
                }
            }
            movieCount++;
        }

        // Before dividing, check if cluster size is zero to avoid division by zero.
        if (cluster.size() == 0) {
            //cout << "Warning: Cluster size is 0!" << endl;
            return totals;  // Return the totals as it is (which are all zeros in this case).
        }
        //calculate the average sums
        for (int i = 0; i < 25; ++i) {
            totals[i] /= cluster.size();
        }
        return totals;
    }

    //method to get centroid closest to mean of cluster
    Movie KmeansPthread::getCentroid(std::vector<Movie>& cluster, vector<float> mean) {
        if (cluster.empty()) {
            cout << "Warning: Cluster is empty!" << endl;
            // Consider handling this appropriately, maybe return an empty Movie or handle it some other way.
            // For now, let's just return the first movie in the cluster for simplicity.
        }

        //initialize global difference and centroid to return
        Movie centroid = cluster[0];
        float diff = 0;
        for (int i = 0; i < 23; ++i) {
            if (i != GROSS) {
                if (i == 24) { // Special debugging for index 24
                    // Checking the size of attr vector in centroid
                    if (centroid.getAttributes().size() <= 24) {
                        // You can handle this error in some specific way, like skipping this iteration or providing some default value
                        return centroid;  // return the current centroid as a placeholder for now
                    }
                    float tempCentroidValue = centroid.getAttributes()[24];   
                    float tempMeanValue = mean[i];
                }
                float tempDiff = centroid[i] - mean[i];
                diff += powf(tempDiff, 2);          
            }
        }
        diff = sqrtf(diff);
        int movieCount = 0; // To keep track of which movie we're processing
        //loop through and find county closest to mean
        for (Movie c : cluster) {
            float local = 0;
            for (int i = 0; i < 23; ++i) {
                if (c.getAttributes().size() <= i) {
                    cout << "Error: Movie " << movieCount << " doesn't have data at index " << i << ". The size of movie's attributes is: " << c.getAttributes().size() << endl;
                    continue;  // Skip this iteration
                }
                if (i != GROSS) local += powf(c[i] - mean[i], 2);
            }
            local = sqrtf(local);

            if (local < diff) {
                diff = local;
                centroid = c;
            }
            movieCount++;
        }
        return centroid;
    }

    //method to setup centroids
    bool KmeansPthread::setupCentroids() {
       

        //cout << "Calculating mean for cluster1..." << endl;
        auto m1 = mean(cluster1);
        
   

        ////get the centroids of each initialized clusters
        Movie c1 = getCentroid(cluster1, mean(cluster1));
        Movie c2 = getCentroid(cluster2, mean(cluster2));

        Movie c3 = getCentroid(cluster3, mean(cluster3));
        cout << "Size of cluster1: " << cluster1.size() << endl;
        cout << "Size of cluster2: " << cluster2.size() << endl;
        cout << "Size of cluster3: " << cluster3.size() << endl;
        cout << "Size of current: " << current.size() << endl;

        //if current and last are the same then return
        if (current[0] == c1 && current[1] == c2 && current[2] == c3) return false;

        //otherwise clear the clusters and push back the clusters
        current[0] = c1;
        current[1] = c2;
        current[2] = c3;
        cluster1.clear();
        cluster2.clear();
        cluster3.clear();
        cluster1.push_back(c1);
        cluster2.push_back(c2);
        cluster3.push_back(c3);
        return true;
    }

    //method to make the clusters
    //more faster
    void KmeansPthread::cluster() {
        int max_iterations = 15;
        int iterations = 0;

        while (iterations < max_iterations) {
            // Assuming you have access to a thread-safe data structure or a way to collect results from threads
            std::vector<std::vector<Movie>> threadwiseClusters(3 * std::thread::hardware_concurrency()); // For each thread: cluster1, cluster2, cluster3

            std::vector<std::thread> threads;
            int movies_per_thread = all.size() / std::thread::hardware_concurrency();

            // Spawn threads
            for (int t = 0; t < std::thread::hardware_concurrency(); ++t) {
                threads.emplace_back([&, t]() {
                    int start_idx = t * movies_per_thread;
                    int end_idx = (t == std::thread::hardware_concurrency() - 1) ? all.size() : start_idx + movies_per_thread;

                    for (int i = start_idx; i < end_idx; ++i) {
                        Movie& movie = all[i];
                        //find which cluster the movie belongs to based on the current centroids
                        if (distanceFromCluster(movie, cluster1) < std::min(distanceFromCluster(movie, cluster2), distanceFromCluster(movie, cluster3))) {
                            threadwiseClusters[3 * t].push_back(movie);
                        }
                        else if (distanceFromCluster(movie, cluster2) < distanceFromCluster(movie, cluster3)) {
                            threadwiseClusters[3 * t + 1].push_back(movie);
                        }
                        else {
                            threadwiseClusters[3 * t + 2].push_back(movie);
                        }
                    }
                    });
            }

            // Join threads
            for (auto& t : threads) {
                t.join();
            }

            // Merge clusters from all threads
            cluster1.clear();
            cluster2.clear();
            cluster3.clear();
            for (int t = 0; t < std::thread::hardware_concurrency(); ++t) {
                cluster1.insert(cluster1.end(), threadwiseClusters[3 * t].begin(), threadwiseClusters[3 * t].end());
                cluster2.insert(cluster2.end(), threadwiseClusters[3 * t + 1].begin(), threadwiseClusters[3 * t + 1].end());
                cluster3.insert(cluster3.end(), threadwiseClusters[3 * t + 2].begin(), threadwiseClusters[3 * t + 2].end());
            }

            // Recalculate centroids
            if (!setupCentroids()) {
                break;  // Break if centroids do not change
            }
            iterations++;
        }
    }
    /* more correct
    void KmeansPthread::cluster() {
    int count = 0;
    auto processCluster = [&](std::vector<Movie>& cluster) {
        while (setupCentroids()) {
            mtx.lock();
            count++;
            
            mtx.unlock();
            // your clustering logic here
            //go through all the data set
            for (Movie c : all) {
                //add to closest cluster
                addToClosest(c);
            }
            if (count >= 10) break;
        }
    };
    
    std::thread t1(processCluster, std::ref(cluster1));
    std::thread t2(processCluster, std::ref(cluster2));
    std::thread t3(processCluster, std::ref(cluster3));
    
    t1.join();
    t2.join();
    t3.join();
    
    
}
    */
    //method to get the distance from a point to rest of cluster
    float KmeansPthread::avgDistance(vector<Movie>& cluster, int index) {
        //cumilate euclidean distance
        float total = 0;
        for (int i = 0; i < cluster.size(); ++i) {
            if (i != index) {
                total += cluster[index] - cluster[i];
            }
        }
        //avg distance from a point to cluster
        float avg = total / (cluster.size() - 1);
        return avg;
    }

    //method to find distance from cluster from a point
    float KmeansPthread::distanceFromCluster(Movie& c, vector<Movie>& cluster) {
        //cumilate distance
        float distance = 0;
        for (Movie& a : cluster) {
            distance += c - a;
        }
        //return distance
        return distance;
    }

    //method to return silhoute value
    float KmeansPthread::silh(vector<Movie>& a, vector<Movie>& b, int index) {
        float aval = avgDistance(a, index);
        float bval = distanceFromCluster(a[index], b);
        float sil = (bval - aval) / max(bval, aval);
        return sil;
    }

    //method to print the silhoute for each cluster
    void KmeansPthread::printSil() {
        //find the value for cluster 1
        float sil = 0;
        for (int i = 0; i < cluster1.size(); ++i) {
            if (distanceFromCluster(cluster1[i], cluster2) < distanceFromCluster(cluster1[i], cluster3)) {
                sil += silh(cluster1, cluster2, i);
            }
            else {
                sil += silh(cluster1, cluster3, i);
            }
        }
        float avsil = sil / cluster1.size();
        
         //find the value for cluster 2
        sil = 0;
        for (int i = 0; i < cluster2.size(); ++i) {
            if (distanceFromCluster(cluster2[i], cluster3) < distanceFromCluster(cluster2[i], cluster1)) {
                sil += silh(cluster2, cluster3, i);
            }
            else {
                sil += silh(cluster2, cluster1, i);
            }
        }
        avsil = sil / cluster2.size();
        
        //find the value for cluster 3
        sil = 0;
        for (int i = 0; i < cluster3.size(); ++i) {
            if (distanceFromCluster(cluster3[i], cluster2) < distanceFromCluster(cluster3[i], cluster1)) {
                sil += silh(cluster3, cluster2, i);
            }
            else {
                sil += silh(cluster3, cluster1, i);
            }
        }
        avsil = sil / cluster3.size();
        

    }
}