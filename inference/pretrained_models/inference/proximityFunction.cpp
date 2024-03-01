#include "../header_files/proximityFunction.h"
using namespace std;

double normal(double x, double y, double scale) {
    double distance = std::abs(x - y);
    double result = std::exp(-distance / scale);
    return result;
}

double uniform(double x, double y, double scale) {
    double distance = std::abs(x - y);
    if (distance < scale) {
        return 1.0;
    } else {
        return 0.0;
    }
}

double weibull(double x, double y, double scale, double shape) {
    double distance = std::abs(x - y);
    double result = std::exp(-std::pow(distance / scale, shape));
    return result;
}

double checkbox(double x, double y, double scale) {
    return 1e6;
}

double beta(double x, double y, double scale, double shape1, double shape2) {
    double distance = std::abs(x - y);
    double result = std::pow(distance, shape1 - 1) * std::pow(scale - distance, shape2 - 1);
    return result;
}