#include "MLP.h"

// Konstruktor
MLP::MLP(const std::vector<int>& topology, int seed)
{
	this->input_size = topology[0];
	this->num_classes = topology.back();

	for (int i = 0; i < topology.size() - 1; i++) {
		// Tworzenie nowej wartwy na GPU
		layers.emplace_back(new Layer(topology[i], topology[i + 1], seed + i));

		// Rezerwacja miejsca na GPU
		double* d_temp_out;
		cudaMalloc(&d_temp_out, topology[i + 1] * sizeof(double));
		layers_outputs.push_back(d_temp_out);
	}
    cudaMalloc(&d_initial_input, input_size * sizeof(double));
    cudaMalloc(&d_target, num_classes * sizeof(double));
}

int MLP::predict(double* input) {

	// Alkoacja dla obrazka wejściowego na GPU
	size_t input_bytes = input_size * sizeof(double);

	// Kopiowanie obrazu z RAM do VRAM
	cudaMemcpy(d_initial_input, input, input_bytes, cudaMemcpyHostToDevice);

    double* current_input = d_initial_input;

	//forward pass
	for (int i = 0; i < layers.size(); ++i) {
		layers[i]->forwardGPU(current_input, layers_outputs[i]);
		current_input = layers_outputs[i];
	}
	std::vector<double> h_results(num_classes);
    cudaMemcpy(h_results.data(), layers_outputs.back(), num_classes * sizeof(double), cudaMemcpyDeviceToHost);


    int best_class = 0;
    double max_val = h_results[0];
    for (int i = 1; i < num_classes; ++i) {
        if (h_results[i] > max_val) {
            max_val = h_results[i];
            best_class = i;
        }
    }
    return best_class;
}

MLP::~MLP() {
	for (int i = 0; i < layers.size(); ++i)
	{
		delete layers[i]; // usunięcie pamięci z VRAM dla wag
		cudaFree(layers_outputs[i]); // zwolnienie bufora pośredniego
	}
    cudaFree(d_initial_input);
    cudaFree(d_target);
}

__global__ void calculateOutputDelta(double* d_output, double* d_target, double* d_deltas, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		double out = d_output[i];
		double target = d_target[i];

        d_deltas[i] = d_target[i] - d_output[i];
	}
}

void MLP::train(double* h_input, int label, double learning_rate) {
    // 1. Forward Pass
    predict(h_input);

    // 2. Przygotowanie targetu - OK
    std::vector<double> h_target(num_classes, 0.0);
    h_target[label] = 1.0;
    cudaMemcpy(d_target, h_target.data(), num_classes * sizeof(double), cudaMemcpyHostToDevice);

    int last = layers.size() - 1;
    int threads = 256;
    int blocks = (num_classes + threads - 1) / threads;

    // 3. Delta wyjściowa (Błąd na samym końcu sieci)
    calculateOutputDelta << <blocks, threads >> > (layers_outputs.back(), d_target, layers[last]->d_deltas, num_classes);

    // 4. Backward Pass - Propagujemy błąd od końca do początku
    for (int i = last - 1; i >= 0; --i) {
        layers[i]->backwardGPU(
            layers[i + 1]->d_weights,
            layers[i + 1]->d_deltas,
            layers_outputs[i],       // Wyjście obecnej warstwy (potrzebne do pochodnej ReLU)
            layers[i + 1]->out_size  // Liczba neuronów w NASTĘPNEJ warstwie
        );
    }

    // 5. Update Weights - Teraz, gdy mamy delty dla KAŻDEJ warstwy, zmieniamy wagi
    double* current_in = d_initial_input;
    for (int i = 0; i <= last; ++i) {
        layers[i]->updateWeightsGPU(current_in, learning_rate);
        current_in = layers_outputs[i]; // Wyjście tej warstwy jest wejściem dla następnej
    }

    // Synchronizacja tylko raz na koniec treningu jednego obrazka
    cudaDeviceSynchronize();
}
