//
//  kmeans.cpp
//  MovieGross
//
//
//  Copyright © 2016 ArsenKevinMD. All rights reserved.
//


#include <stdio.h>
#include "movie.h"
#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iostream>


using namespace std;
using namespace MovieData;

//K-Means Clustering Namespace
namespace KmeansCluster{
    
    //find the closest cluster to add
    void KMeans:: addToClosest(MovieData::Movie&amovie){
        //check if county passed in centroid
        if(current[0] == amovie || current[1] == amovie || current[2] == amovie ) return;
        
        
        //get the distance from the centroid to the county
        float distance1 = distanceFromCluster(amovie,cluster1);
        float distance2 = distanceFromCluster(amovie,cluster2);
        float distance3 = distanceFromCluster(amovie,cluster3);
        
        //check which distance is the smallest
        float smallest = distance1;
        if(smallest > distance2){
            smallest = distance2;
        }
        if(smallest > distance3){
            smallest = distance3;
        }
        
        //based on the smallest distance add to cluster
        if(smallest == distance1){
            cluster1.push_back(amovie);
        }
        else if(smallest == distance2){
            cluster2.push_back(amovie);
        }
        else{
            cluster3.push_back(amovie);
        }
    }
    
    //method to predict movie gross
    float KMeans::predict(MovieData::Movie aMovie){
        //get the distance from the centroid to the county
        float distance1 = distanceFromCluster(aMovie,cluster1);
        float distance2 = distanceFromCluster(aMovie,cluster2);
        float distance3 = distanceFromCluster(aMovie,cluster3);

        //check which distance is the smallest
        float smallest = distance1;
        if(smallest > distance2){
            smallest = distance2;
        }
        if(smallest > distance3){
            smallest = distance3;
        }
        
        //based on the smallest distance add to cluster
        float total = 0, size = 0;
        
        if(smallest == distance1){
            for(Movie m : cluster1){
                total += m[GROSS];
                size +=  1;
            }
        }
        else if(smallest == distance2){
            for(Movie m : cluster2){
                total += m[GROSS];
                size +=  1;
            }
        }
        else{
            for(Movie m : cluster3){
                total += m[GROSS];
                size +=  1;
            }
        }
        return total/size;
    }
    
    //method to initialize
    void KMeans:: initialize(vector<Movie> movies){
        //initialize random
        srand(time(NULL));
        
        //get three random indexes
        int index1 = rand() % 100;
        int index2 = rand() % 100 + 100;
        int index3 = rand() % 200 + 300;
        
        //get the three random centroids
        Movie first = movies[index1];
        //cout << "Accessed first movie." << endl;
        Movie second = movies[index2];
        //cout << "Accessed second movie." << endl;
        Movie third = movies[index3];
        //cout << "Accessed third movie." << endl;
        //push them into the clusters and setup centroids
        cluster1.push_back(first);
        //cout << "Accessed cluster1.push_back" << endl;
        cluster2.push_back(second);
        //cout << "Accessed cluster2.push_back" << endl;
        cluster3.push_back(third);
        //cout << "Accessed cluster3.push_back" << endl;
        current.push_back(first);
        //cout << "Accessed current.push_back" << endl;
        current.push_back(second);
        //cout << "Accessed current.push_back" << endl;
        current.push_back(third);
        //cout << "Accessed current.push_back" << endl;
        //go through counties and add to each cluster
        for(Movie c : movies){
            //cout << "Movie c : movies" << endl;
            addToClosest(c);
            //cout << "addToClosest(c);" << endl;
            all.push_back(c);
            //cout << "all.push_back(c);" << endl;
        }
        //cout << "  for(Movie c : movies) done" << endl;
    }
    
    //method to get the mean of a cluster
    vector<float> KMeans::mean(std::vector<Movie>& cluster) {
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

            //cout << "Finished processing movie " << movieCount << endl;
            movieCount++;
        }
        //cout << "Finished tallying the total sum." << endl;

        // Before dividing, check if cluster size is zero to avoid division by zero.
        if (cluster.size() == 0) {
            //cout << "Warning: Cluster size is 0!" << endl;
            return totals;  // Return the totals as it is (which are all zeros in this case).
        }

        //cout << "Starting to calculate average sums..." << endl;
        //calculate the average sums
        for (int i = 0; i < 25; ++i) {
            totals[i] /= cluster.size();
        }
        //cout << "Finished calculating average sums." << endl;

        //cout << "Exiting mean() function" << endl;
        return totals;
    }
    
    //method to get centroid closest to mean of cluster
    Movie KMeans::getCentroid(std::vector<Movie>& cluster, vector<float> mean) {
        //cout << "Inside getCentroid() function" << endl;

        if (cluster.empty()) {
            cout << "Warning: Cluster is empty!" << endl;
            // Consider handling this appropriately, maybe return an empty Movie or handle it some other way.
            // For now, let's just return the first movie in the cluster for simplicity.
        }

        //initialize global difference and centroid to return
        Movie centroid = cluster[0];
        float diff = 0;
        //cout << "Calculating initial difference..." << endl;
        //cout << "Size of the centroid attributes: " << centroid.getSize() << endl;
        for (int i = 0; i < 23; ++i) {
            if (i != GROSS) {
                //cout << "Processing index " << i << endl;

                if (i == 24) { // Special debugging for index 24
                    //cout << "Special check for index 24" << endl;
                    //cout << "Trying to access centroid at index 24..." << endl;

                    // Checking the size of attr vector in centroid
                    if (centroid.getAttributes().size() <= 24) {
                        //cout << "Error: Centroid doesn't have data at index 24. The size of centroid's attributes is: " << centroid.getAttributes().size() << endl;
                        // You can handle this error in some specific way, like skipping this iteration or providing some default value
                        return centroid;  // return the current centroid as a placeholder for now
                    }
                    float tempCentroidValue = centroid.getAttributes()[24];
                    //cout << "Successfully accessed centroid value at index 24: " << tempCentroidValue << endl;

                    //cout << "Trying to access mean at index 24..." << endl;
                    float tempMeanValue = mean[i];
                    //cout << "Successfully accessed mean value at index 24: " << tempMeanValue << endl;
                }

                //cout << "Centroid value at index " << i << ": " << centroid.getAttributes()[i] << endl;
                //cout << "Mean value at index " << i << ": " << mean[i] << endl;

                float tempDiff = centroid[i] - mean[i];
                //cout << "Difference at index " << i << ": " << tempDiff << endl;
                diff += powf(tempDiff, 2);
                //cout << "Accumulated square difference after index " << i << ": " << diff << endl;
            }
        }
        //cout << "Size of the centroid attributes: " << centroid.getSize() << endl;
        diff = sqrtf(diff);
        //cout << "Initial difference calculated: " << diff << endl;

        int movieCount = 0; // To keep track of which movie we're processing
        //loop through and find county closest to mean
        for (Movie c : cluster) {
            //cout << "Processing movie " << movieCount << " for centroid calculation." << endl;

            float local = 0;
            for (int i = 0; i < 23; ++i) {
                if (c.getAttributes().size() <= i) {
                    cout << "Error: Movie " << movieCount << " doesn't have data at index " << i << ". The size of movie's attributes is: " << c.getAttributes().size() << endl;
                    continue;  // Skip this iteration
                }
                if (i != GROSS) local += powf(c[i] - mean[i], 2);
            }
            local = sqrtf(local);
            //cout << "Local difference for movie " << movieCount << ": " << local << endl;

            if (local < diff) {
                //cout << "Found a closer movie at index " << movieCount << " with difference: " << local << endl;
                diff = local;
                centroid = c;
            }

            movieCount++;
        }

        //cout << "Exiting getCentroid() function" << endl;
        return centroid;
    }
    
    //method to setup centroids
    bool KMeans::setupCentroids(){
        //cout << "Inside setupCentroids" << endl;

        //cout << "Calculating mean for cluster1..." << endl;
        auto m1 = mean(cluster1);
        //cout << "Calculated mean for cluster1. Value: [";
        for (size_t i = 0; i < m1.size(); ++i) {
            //cout << m1[i];
            if (i < m1.size() - 1) {
                //cout << ", ";
            }
        }
        //cout << "]" << endl;

       /* cout << "Calculating c1 using the mean..." << endl;
        Movie c1 = getCentroid(cluster1, m1);
        cout << "Calculated c1." << endl;*/

        ////get the centroids of each initialized clusters
        //cout << "Calculating c1..." << endl;
        Movie c1 = getCentroid(cluster1, mean(cluster1));
        //cout << "Calculated c1." << endl;

        //cout << "Calculating c2..." << endl;
        Movie c2 = getCentroid(cluster2, mean(cluster2));
        //cout << "Calculated c2." << endl;

        //cout << "Calculating c3..." << endl;
        Movie c3 = getCentroid(cluster3, mean(cluster3));
        //cout << "Calculated c3." << endl;
        cout << "Size of cluster1: " << cluster1.size() << endl;
        cout << "Size of cluster2: " << cluster2.size() << endl;
        cout << "Size of cluster3: " << cluster3.size() << endl;
        cout << "Size of current: " << current.size() << endl;
        
        //if current and last are the same then return
        if(current[0] == c1 && current[1] == c2 && current[2] == c3) return false;
        
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
    void KMeans:: cluster(){
        int count = 0;
        //cout << "  void KMeans:: cluster()" << endl;
        //while the centroids update
        //cout << "while(setupCentroids())" << endl;
        while(setupCentroids()){
            //cout << "while(setupCentroids()) in" << endl;
            count++;
            //cout << "clustering..." << endl;
            //cout << "while(setupCentroids()) in2" << endl;
            //go through all the data set
            for(Movie c : all){
                //add to closest cluster
                addToClosest(c);
            }
            if(count >= 10) break;
            
        }
    }
    
    //method to get the distance from a point to rest of cluster
    float KMeans:: avgDistance(vector<Movie>&cluster,int index){
        //cumilate euclidean distance
        float total = 0;
        for(int i = 0; i < cluster.size(); ++i){
            if(i != index){
                total += cluster[index]-cluster[i];
            }
        }
        //avg distance from a point to cluster
        float avg = total/(cluster.size()-1);
        return avg;
    }
    
    //method to find distance from cluster from a point
    float KMeans:: distanceFromCluster(Movie&c,vector<Movie>&cluster){
        //cumilate distance
        float distance = 0;
        for(Movie& a : cluster){
            distance += c-a;
        }
        //return distance
        return distance;
    }
    
    //method to return silhoute value
    float KMeans:: silh(vector<Movie>&a,vector<Movie>&b,int index){
        float aval = avgDistance(a, index);
        float bval = distanceFromCluster(a[index], b);
        float sil = (bval - aval)/max(bval,aval);
        return sil;
    }
    
    //method to print the silhoute for each cluster
    void KMeans:: printSil(){
        //find the value for cluster 1
        float sil = 0;
        for(int i = 0; i < cluster1.size(); ++i){
            if(distanceFromCluster(cluster1[i],cluster2) < distanceFromCluster(cluster1[i],cluster3)){
                sil += silh(cluster1,cluster2,i);
            }else{
                sil += silh(cluster1,cluster3,i);
            }
        }
        float avsil = sil/cluster1.size();
       // cout << "cluster 1 similarity: " << avsil << endl;
        //find the value for cluster 2
        sil = 0;
        for(int i = 0; i < cluster2.size(); ++i){
            if(distanceFromCluster(cluster2[i],cluster3) < distanceFromCluster(cluster2[i],cluster1)){
                sil += silh(cluster2,cluster3,i);
            }else{
                sil += silh(cluster2,cluster1,i);
            }
        }
        avsil = sil/cluster2.size();
        //cout << "cluster 2 similarity: " << avsil << endl;
        //find the value for cluster 3
        sil = 0;
        for(int i = 0; i < cluster3.size(); ++i){
            if(distanceFromCluster(cluster3[i],cluster2) < distanceFromCluster(cluster3[i],cluster1)){
                sil += silh(cluster3,cluster2,i);
            }else{
                sil += silh(cluster3,cluster1,i);
            }
        }
        avsil = sil/cluster3.size();
        //cout << "cluster 3 similarity: " << avsil << endl;
        
    }
}
