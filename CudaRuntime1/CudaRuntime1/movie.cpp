#include "movie.h"
#include <math.h>

using namespace CsvProc;

namespace MovieData {

    Movie::Movie(float data[ATTR_SIZE]) {
        for (int i = 0; i < ATTR_SIZE; ++i) {
            attr[i] = data[i];
        }
    }

    Movie::Movie() {
        for (int i = 0; i < ATTR_SIZE; ++i) {
            attr[i] = 0.0f;
        }
    }

    void Movie::normalize(Csv& csv) {
        for (int i = 0; i < ATTR_SIZE; ++i) {
            attr[i] = (attr[i] - csv.MIN[i]) / (csv.MAX[i] - csv.MIN[i]);
        }
    }

    float Movie::operator[](int index) {
        return attr[index];
    }

    float Movie::operator-(Movie& aMovie) {
        float dist = 0;
        for (int i = 0; i < ATTR_SIZE; ++i) {
            if (i != GROSS) dist += powf(attr[i] - aMovie[i], 2);
        }
        return sqrtf(dist);
    }

    bool Movie::operator==(Movie& aMovie) {
        for (int i = 0; i < ATTR_SIZE; ++i) {
            if (attr[i] != aMovie[i]) return false;
        }
        return true;
    }

    int Movie::getSize() const {
        return ATTR_SIZE;
    }

    Movie::Movie(const std::vector<float>& data) {
        for (int i = 0; i < ATTR_SIZE && i < data.size(); ++i) {
            attr[i] = data[i];
        }
    }
    /*float Movie::operator[](int index) const {
        return attr[index];
    }*/
    
}
