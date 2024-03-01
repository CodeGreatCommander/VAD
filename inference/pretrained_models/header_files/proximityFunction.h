#pragma once

#include <cmath>
double normal(double x, double y, double scale = 1);
double uniform(double x, double y, double scale = 1e9);
double weibull(double x, double y, double scale = 1.0, double shape = 1.0);
double checkbox(double x, double y, double scale = 1.0);
double beta(double x, double y, double scale = 1.0, double shape1 = 1.0, double shape2 = 1.0);