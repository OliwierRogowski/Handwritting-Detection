#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class Layer {
public:
	Eigen::MatrixXd h_weights; // Adres danych do CPU
	double* d_weights;	// Adres danych do GPU
	Eigen::VectorXd h_biases;
	double* d_biases;
	int out_size;
	int in_size;
	double* d_deltas;
	Layer(int in_size, int out_size, int seed);
	~Layer(); // destruktor cudaFree

	void updateWeightsGPU(double* d_input, double learning_rate);

	void backwardGPU(double* d_next_weights, double* d_next_deltas, double* d_current_output, int next_out_size);

	void forwardGPU(double* d_input, double* d_output);
};
