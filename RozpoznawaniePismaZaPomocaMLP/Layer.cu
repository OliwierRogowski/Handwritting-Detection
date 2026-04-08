#include <iostream>
#include "Layer.h"
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Dense>

// KERNELE CUDA
/*
* Kernele cuda to funkcje w c++ które pozwalają na programowanie na kartach graficznych NVIDIA. Taka funkcja może przetworzyć równolegle tysiące danych przez tysiące wqątków na GPU.
* Funkcja napisana na CPU będzie sie wykonywać raz po wywołaniu, natomiast kernel CUDA po uruchomieniu wykonuje sięmasowo na wielu danych jednocześnie.
*/

// Forward Pass: Oblicza wyjście warstwy
// Forward pass bierze sygnały wejściowe i mnozy je przez wagi, dodaje bias i przepuszcza przez funkcję aktywacji aby pokazać to co sieć widzi
__global__ void forwardLayer(double* input, double* output, double* weights, double* d_biases, int in_size, int out_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Identyfiukacja wątków. Każdy wątek musi wiedzieć za ktyóry neuron konkretnie odpowiada i zmienna row oblicza unikalny numer neuronu.
    if (row < out_size) {
        double sum = 0.0;
        // Mnożenie macieży Iloczyn skalarny
        for (int i = 0; i < in_size; ++i) {
            sum += weights[row + i * out_size] * input[i];
        }
        // Dodanie przesunięcia, inaczej obciążenia jednostkowego lub biasu.
        double z = sum + d_biases[row];
        
        // Funkcja Leay ReLU to funkcja aktywacji, Różni sięod ReLU tym żę zamiast zerować wartości ujemne, mnoży je przez małą liczbę w tym przypadku 0.01
        output[row] = (z > 0) ? z : (0.01 * z);
    }
}

// Backward Pass: Oblicza błąd (deltę) płynący do poprzedniej warstwy
// Backward Pass słóży do wyciągania wniosków po predykcji z forword pass. To tutaj wagi są aktualizowane i dzieje się uczenie
__global__ void backwardLayer(double* d_weights, double* d_next_deltas, double* d_output_current, double* d_deltas_out, int in_size, int out_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Indeks neuronu w OBECNEJ warstwie (wejście)

    if (i < in_size) {
        double error = 0.0;
        // Akumulacja błędu, czyli iloczyn wag i delt
        for (int j = 0; j < out_size; ++j) {
            error += d_weights[j + i * out_size] * d_next_deltas[j];
        }

        // Pochodna Leaky ReLU
        // Pochodna w tym miejscu pozwala na przejscie błędu dlaej przez neuron.
        // Pochodna mówi nam jak zmiana wejścia wpływa na wyjście a więc
        // Jeżeli neuron byl aktywny czyli z > 0 to pochnodna wyniesie jeden a więc błąd przekazujemy cały, ponieważ error * 1
        // Jeśli natomiast neuron był nieaktywny czyli z rowne badz mniejsze zeru to podchodna funkcji aktywacji jest rowna 0.01 i przekazujemy maly ulamek bledu
        d_deltas_out[i] = (d_output_current[i] > 0) ? error : (error * 0.01);
    }
}

// Update Weights: Aktualizuje wagi i biasy na podstawie policzonych delt
// Ten kernel pozwala na zmiane sieci. CZyli na podstawie forwardLayer oraz backwardLayer ten kernel poprawia wyniki aby przy następnym przejściu (epoch) wynik był lepszy niż poprzedni
__global__ void updateWeights(double* d_weights, double* d_biases, double* d_input, double* d_deltas, double learning_rate, int in_size, int out_size) {
    // Praca w dwóch wymiarach 
    // Ten kod uruchamia osobny wątek dla każdej pojedynczej wagi w warstwie.
    int row = blockIdx.x * blockDim.x + threadIdx.x; // neuron (wiersz)
    int col = blockIdx.y * blockDim.y + threadIdx.y; // wejście (kolumna)

    if (row < out_size && col < in_size) {
        int weight_idx = row + col * out_size;

        // W = W + LR * delta * input
        // To jest formuła aktualizacji wagi
        d_weights[weight_idx] += learning_rate * d_deltas[row] * d_input[col];

        // Bias dodajemy tylko raz aby zapobiec zniszczeniu obliczen. Bias jest aktualizowany na podstawie bledu samego neuronu a nie zalezy od sygnalu wejsciowego to też dlatego bias jest aktualizowany raz dla kazdego neuronu
        if (col == 0) {
            d_biases[row] += learning_rate * d_deltas[row];
        }
    }
}

// --- METODY KLASY LAYER ---

Layer::Layer(int in_size, int out_size, int seed) : in_size(in_size), out_size(out_size) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // 1. Inicjalizacja na CPU
    h_weights.resize(out_size, in_size);
    h_biases = Eigen::VectorXd::Zero(out_size);

    double scale = 0.1;
    for (int i = 0; i < h_weights.rows(); ++i) {
        for (int j = 0; j < h_weights.cols(); ++j) {
            h_weights(i, j) = dist(gen) * scale;
        }
        h_biases[i] = 0.01; // Mały bias na start
    }

    // 2. Alokacja Wag na GPU
    size_t bytes_w = in_size * out_size * sizeof(double);
    cudaMalloc(&d_weights, bytes_w);
    cudaMemcpy(d_weights, h_weights.data(), bytes_w, cudaMemcpyHostToDevice);

    // 3. Alokacja Biasów na GPU
    size_t bytes_b = out_size * sizeof(double);
    cudaMalloc(&d_biases, bytes_b);
    cudaMemcpy(d_biases, h_biases.data(), bytes_b, cudaMemcpyHostToDevice);

    // 4. Alokacja Delt na GPU
    // Rozmiar in_size, bo delty są przekazywane "wstecz" do wejść warstwy
    size_t bytes_d = in_size * sizeof(double);
    cudaError_t err_d = cudaMalloc(&d_deltas, bytes_d);
    if (err_d != cudaSuccess) {
        std::cerr << "CUDA Malloc deltas failed: " << cudaGetErrorString(err_d) << std::endl;
    }
    cudaMemset(d_deltas, 0, bytes_d);
}

Layer::~Layer() {
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_deltas);
}

void Layer::forwardGPU(double* input, double* output) {
    int threadsPerBlock = 256;
    int blockPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;

    forwardLayer << <blockPerGrid, threadsPerBlock >> > (input, output, d_weights, d_biases, in_size, out_size);
    cudaDeviceSynchronize();
}

void Layer::backwardGPU(double* d_next_weights, double* d_next_deltas, double* d_current_output, int next_out_size) {
    int threads = 256;
    int blocks = (in_size + threads - 1) / threads;

    // Obliczamy delty dla TEJ warstwy (in_size) na podstawie warstwy NASTĘPNEJ (next_out_size)
    backwardLayer << <blocks, threads >> > (d_next_weights, d_next_deltas, d_current_output, d_deltas, in_size, next_out_size);
    cudaDeviceSynchronize();
}

void Layer::updateWeightsGPU(double* d_input, double learning_rate) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_size + 15) / 16, (in_size + 15) / 16);

    updateWeights << <numBlocks, threadsPerBlock >> > (d_weights, d_biases, d_input, d_deltas, learning_rate, in_size, out_size);
    cudaDeviceSynchronize();
}
