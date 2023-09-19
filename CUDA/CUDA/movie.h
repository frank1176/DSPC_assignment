#ifndef movie_h
#define movie_h

#include "csv.h"

//Global constants
const int GROSS = 7;
const int YEAR = 19;
const int ATTR_SIZE = 20; // Assuming you have 20 attributes

//namespace for movie object
namespace MovieData {
    //data structure to process movie data
    class Movie {
    private:
        float attr[ATTR_SIZE];
        std::vector<float> attributes_;
    public:
        //constructors
        Movie();  // Default constructor
        Movie(float data[ATTR_SIZE]); // Parameterized constructor

        Movie(const std::vector<float>& attributes);

        //method to normalize the data
        void normalize(CsvProc::Csv& csv);

        //Operator to return attribute value
        float operator[](int index);

        //Operator to return eucledian distance
        float operator-(Movie& aMovie);

        //Operator to return whether two Movies are the same
        bool operator==(Movie& aMovie);

        int getSize() const;

       
    };
}

#endif /* movie_h */
