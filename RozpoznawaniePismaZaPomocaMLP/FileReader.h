#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>

class FileReader {
private:
	int num_images;
	int image_size;

	// Data for CPU (Host)
	// It is list of vectors of images
	std::vector<std::vector<double>> h_images;
	std::vector<int> h_labels;

	int reverseInt(int i);

public:
	FileReader() : num_images(0), image_size(784) {} // Pictures are 28x28 so it's 784 pixels

	bool loadDataset(const std::string& image_path, const std::string& label_path);
	
	const std::vector<double>& getImage(int index) const { return h_images[index]; }
	int getLabel(int index) const { return h_labels[index]; }
	int getNumImages() const { return num_images; }


	void copyImageToGPU(int index, double* d_target) const {
		size_t bytes = image_size * sizeof(double);
		cudaMemcpy(d_target, h_images[index].data(), bytes, cudaMemcpyHostToDevice);
	}

	void copyAllToGPU(double** d_all_images) const;
};
