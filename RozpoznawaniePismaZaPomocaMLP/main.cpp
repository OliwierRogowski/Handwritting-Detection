#include <iostream>
#include <vector>
#include <iomanip>
#include "MLP.h"
#include "FileReader.h"
#include <opencv2/opencv.hpp>

// Funkcja do przygotowania własnego zdjęcia
void testOwnJpg(MLP& network, const std::string& filePath) {
    // wczytanie obrazu
    cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Nie znaleziono pliku: " << filePath << std::endl;
        return;
    }

    cv::Mat binaryImg;
    // Automatyczne dobieranie progu jasności aby oddzielić cyfrę od tła
    cv::threshold(img, binaryImg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // Kolory muszą być odwrócone, ponieważ w dbazie danych MNIST mamy czarne tło oraz białą cyfrę

    std::vector<cv::Point> points;
    cv::findNonZero(binaryImg, points);

    cv::Mat finalResized;

    if (!points.empty()) {
        // Wycinanie cyfry
        cv::Rect bbox = cv::boundingRect(points);

        int pad = 20;
        bbox.x = std::max(0, bbox.x - pad);
        bbox.y = std::max(0, bbox.y - pad);
        bbox.width = std::min(binaryImg.cols - bbox.x, bbox.width + 2 * pad);
        bbox.height = std::min(binaryImg.rows - bbox.y, bbox.height + 2 * pad);

        cv::Mat cropped = binaryImg(bbox);

        // skalowanie obrazu do rozmiaru 28x28
        cv::resize(cropped, finalResized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
    }
    else {
        finalResized = cv::Mat::zeros(28, 28, CV_8UC1);
    }

    // Dylatacja czyli pogrubianie linie
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(finalResized, finalResized, kernel);

    // Sprawdzenie co widzi sieć, jeżeli liczba jest przsunieta do rogu bądź jest po prostu nie czytelna, sieć jej nie odrużni
    cv::imshow("Co widzi siec", finalResized);
    cv::waitKey(500); 
    cv::destroyAllWindows();

    // przygotowanie danych do sieci
    std::vector<double> input;
    input.reserve(784);
    for (int i = 0; i < 28 * 28; ++i) {
        input.push_back(static_cast<double>(finalResized.data[i]) / 255.0);
    }

    // Predykcja sieci. Do sieci trafiaja przetworzone dane obrazu po treningu z zapisanymi wagami aby model mógł całkowicie rozpoznać odręcznącyfre od 0 do 9
    int result = network.predict(input.data());

    std::cout << "-----------------------------" << std::endl;
    std::cout << "WYNIK DLA: " << filePath << " -> " << result << std::endl;
    std::cout << "-----------------------------" << std::endl;
}
int main() {
    FileReader reader;
    // Ścieżka do danych testowych bazy minst
    std::string imgPath = "C:/Users/orogo/Desktop/Projekty/RozpoznawaniePismaZaPomocaMLP/archive/train-images.idx3-ubyte";
    std::string lblPath = "C:/Users/orogo/Desktop/Projekty/RozpoznawaniePismaZaPomocaMLP/archive/train-labels.idx1-ubyte";

    std::cout << "[INFO] Wczytywanie bazy MNIST..." << std::endl;
    if (!reader.loadDataset(imgPath, lblPath)) {
        std::cerr << "[ERROR] Nie udalo sie znalezc plikow w: " << imgPath << std::endl;
        return -1;
    }

    // 2. Konfiguracja Sieci: 784 wejscia -> 128 ukryte -> 10 wyjscia
    std::vector<int> topology = { 784, 128, 10 };
    double learningRate = 0.001; // współczynnik uczenia się
    int epochs = 15; // liczba przejść sieci
    
    MLP network(topology, 42); // 42 to seed dla losowosci wag

    std::cout << "[INFO] Rozpoczynanie treningu..." << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // 3. Glowna petla treningowa
    for (int e = 0; e < epochs; ++e) {
        int correct = 0; 
        double epochLoss = 0.0;

        for (int i = 0; i < reader.getNumImages(); ++i) {
            // Pobieramy dane z readera
            const std::vector<double>& input = reader.getImage(i);
            int label = reader.getLabel(i);

            // Najpierw sprawdzamy co siec mysli (do statystyk)
            int prediction = network.predict((double*)input.data());
            if (prediction == label) correct++;

            // Trening (Forward + Backward + Update)
            network.train((double*)input.data(), label, learningRate);

            // Logowanie postepu co 1000 obrazkow
            if (i % 1000 == 0 && i > 0) {
                double accuracy = (double)correct / (i + 1) * 100.0;
                std::cout << "Epoch [" << e + 1 << "/" << epochs << "] "
                          << "Progress: " << std::fixed << std::setprecision(1) 
                          << (double)i / reader.getNumImages() * 100.0 << "% "
                          << "| Accuracy: " << accuracy << "%   \r" << std::flush;
            }
        }

        // Podsumowanie epoki
        double finalAccuracy = (double)correct / reader.getNumImages() * 100.0;
        std::cout << "\n[EPOCH " << e + 1 << "] Koniec. Srednia celnosc: " << finalAccuracy << "%" << std::endl;
        
        learningRate *= 0.05; 
    }

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "[SUCCESS] Trening zakonczony. " << std::endl;

    // Własne obrazy
    testOwnJpg(network, "C:/Users/orogo/Pictures/tst/dwa.jpg");
    testOwnJpg(network, "C:/Users/orogo/Pictures/tst/osiem.jpg");
    testOwnJpg(network, "C:/Users/orogo/Pictures/tst/szesc.jpg");
    testOwnJpg(network, "C:/Users/orogo/Pictures/tst/zero.jpg");
    return 0;

}
