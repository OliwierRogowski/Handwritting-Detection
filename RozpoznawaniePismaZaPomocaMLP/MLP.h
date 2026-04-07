#pragma once
#include <vector>
#include <string>
#include "Layer.h"

class MLP {
private: 
	double* d_target;
	std::vector<Layer*> layers; // kontener na warstwy GPU
	int input_size;
	int num_classes;
	double* d_initial_input;

	// Pomocnicze wskaźniki na pamięć GPU dla danych pośrednich
	std::vector<double*> layers_outputs;

public: 
	MLP(const std::vector<int>& topology, int seed);
	~MLP(); // Destruktor

	void train(double* h_input, int label, double learning_rate);

	// Główna funckja przewidywania
	int predict(double* d_input);

	/*
	void saveModel(const std::string& filename);
	void loadModel(const std::string& filename);*/
};
