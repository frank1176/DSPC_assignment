#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "movie.h"
#include "kmeans.h"

using namespace std;
using namespace MovieData;

namespace KmeansCluster {

    void KMeansOpenMP::addToClosest(Movie& amovie) {
        if (current[0] == amovie || current[1] == amovie || current[2] == amovie) return;

        float distance1 = distanceFromCluster(amovie, cluster1);
        float distance2 = distanceFromCluster(amovie, cluster2);
        float distance3 = distanceFromCluster(amovie, cluster3);

        float smallest = distance1;
        if (smallest > distance2) {
            smallest = distance2;
        }
        if (smallest > distance3) {
            smallest = distance3;
        }

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

    float KMeansOpenMP::predict(Movie aMovie) {
        float distance1 = distanceFromCluster(aMovie, cluster1);
        float distance2 = distanceFromCluster(aMovie, cluster2);
        float distance3 = distanceFromCluster(aMovie, cluster3);

        float smallest = distance1;
        if (smallest > distance2) {
            smallest = distance2;
        }
        if (smallest > distance3) {
            smallest = distance3;
        }

        float total = 0, size = 0;

        if (smallest == distance1) {
    #pragma omp parallel for reduction(+:total, size)
            for (int i = 0; i < cluster1.size(); ++i) {
                total += cluster1[i][GROSS];
                size += 1;
            }
        }
        else if (smallest == distance2) {
    #pragma omp parallel for reduction(+:total, size)
            for (int i = 0; i < cluster2.size(); ++i) {
                total += cluster2[i][GROSS];
                size += 1;
            }
        }
        else {
    #pragma omp parallel for reduction(+:total, size)
            for (int i = 0; i < cluster3.size(); ++i) {
                total += cluster3[i][GROSS];
                size += 1;
            }
        }
        return total / size;
    }

    void KMeansOpenMP::initialize(vector<Movie> movies) {
        srand(time(NULL));

        int index1 = rand() % 100;
        int index2 = rand() % 100 + 100;
        int index3 = rand() % 200 + 300;

        Movie first = movies[index1];
        Movie second = movies[index2];
        Movie third = movies[index3];

        cluster1.push_back(first);
        cluster2.push_back(second);
        cluster3.push_back(third);
        current.push_back(first);
        current.push_back(second);
        current.push_back(third);

    #pragma omp parallel for
        for (int i = 0; i < movies.size(); ++i) {
            addToClosest(movies[i]);
            all.push_back(movies[i]);
        }
    }

    vector<float> KMeansOpenMP::mean(vector<Movie>& cluster) {
        vector<float> totals(25, 0);

    #pragma omp parallel for
        for (int i = 0; i < cluster.size(); ++i) {
            for (int j = 0; j < 23; ++j) {
                if (j != GROSS) {
                    if (cluster[i].getSize() <= j) {
                        continue;
                    }
                    totals[j] += cluster[i][j];
                }
            }
        }

        if (cluster.size() == 0) {
            return totals;
        }

    #pragma omp parallel for
        for (int i = 0; i < 25; ++i) {
            totals[i] /= cluster.size();
        }

        return totals;
    }

    Movie KMeansOpenMP::getCentroid(vector<Movie>& cluster, vector<float> mean) {
        if (cluster.empty()) {
            cout << "Warning: Cluster is empty!" << endl;
            // Handle this case appropriately.
            // For now, return the first movie in the cluster.
            return cluster[0];
        }

        Movie centroid = cluster[0];
        float diff = 0;

        for (int i = 0; i < 23; ++i) {
            if (i != GROSS) {
                float tempDiff = centroid[i] - mean[i];
                diff += powf(tempDiff, 2);
            }
        }

        diff = sqrtf(diff);

        int movieCount = 0;
    #pragma omp parallel for
        for (int i = 0; i < cluster.size(); ++i) {
            float local = 0;
            for (int j = 0; j < 23; ++j) {
                if (cluster[i].getAttributes().size() <= j) {
                    cout << "Error: Movie " << movieCount << " doesn't have data at index " << j << ". The size of movie's attributes is: " << cluster[i].getAttributes().size() << endl;
                    continue;
                }
                if (j != GROSS) local += powf(cluster[i][j] - mean[j], 2);
            }
            local = sqrtf(local);

            if (local < diff) {
                diff = local;
                centroid = cluster[i];
            }
        }

        return centroid;
    }

    bool KMeansOpenMP::setupCentroids() {
        auto m1 = mean(cluster1);
        Movie c1 = getCentroid(cluster1, m1);

        Movie c2 = getCentroid(cluster2, mean(cluster2));
        Movie c3 = getCentroid(cluster3, mean(cluster3));

        if (current[0] == c1 && current[1] == c2 && current[2] == c3) return false;

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

    void KMeansOpenMP::cluster() {
        int count = 0;

        while (setupCentroids()) {
            count++;

    #pragma omp parallel for
            for (int i = 0; i < all.size(); ++i) {
                addToClosest(all[i]);
            }

            if (count >= 10) break;
        }
    }

    float KMeansOpenMP::avgDistance(vector<Movie>& cluster, int index) {
        float total = 0;

    #pragma omp parallel for reduction(+:total)
        for (int i = 0; i < cluster.size(); ++i) {
            if (i != index) {
                total += cluster[index] - cluster[i];
            }
        }

        float avg = total / (cluster.size() - 1);
        return avg;
    }

    float KMeansOpenMP::distanceFromCluster(Movie& c, vector<Movie>& cluster) {
        float distance = 0;

    #pragma omp parallel for reduction(+:distance)
        for (int i = 0; i < cluster.size(); ++i) {
            distance += c - cluster[i];
        }

        return distance;
    }

    float KMeansOpenMP::silh(vector<Movie>& a, vector<Movie>& b, int index) {
        float aval = avgDistance(a, index);
        float bval = distanceFromCluster(a[index], b);
        float sil = (bval - aval) / max(bval, aval);
        return sil;
    }

    void KMeansOpenMP::printSil() {
        float sil = 0;

    #pragma omp parallel for reduction(+:sil)
        for (int i = 0; i < cluster1.size(); ++i) {
            if (distanceFromCluster(cluster1[i], cluster2) < distanceFromCluster(cluster1[i], cluster3)) {
                sil += silh(cluster1, cluster2, i);
            }
            else {
                sil += silh(cluster1, cluster3, i);
            }
        }

        float avsil = sil / cluster1.size();
        cout << "cluster 1 similarity: " << avsil << endl;

        sil = 0;

    #pragma omp parallel for reduction(+:sil)
        for (int i = 0; i < cluster2.size(); ++i) {
            if (distanceFromCluster(cluster2[i], cluster3) < distanceFromCluster(cluster2[i], cluster1)) {
                sil += silh(cluster2, cluster3, i);
            }
            else {
                sil += silh(cluster2, cluster1, i);
            }
        }

        avsil = sil / cluster2.size();
        cout << "cluster 2 similarity: " << avsil << endl;

        sil = 0;

    #pragma omp parallel for reduction(+:sil)
        for (int i = 0; i < cluster3.size(); ++i) {
            if (distanceFromCluster(cluster3[i], cluster2) < distanceFromCluster(cluster3[i], cluster1)) {
                sil += silh(cluster3, cluster2, i);
            }
            else {
                sil += silh(cluster3, cluster1, i);
            }
        }

        avsil = sil / cluster3.size();
        cout << "cluster 3 similarity: " << avsil << endl;
    }
}
