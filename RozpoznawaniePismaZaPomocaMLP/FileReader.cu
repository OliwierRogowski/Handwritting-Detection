#include "FileReader.h"

bool FileReader::loadDataset(const std::string& image_path, const std::string& label_path) {
    std::ifstream img_file(image_path, std::ios::binary);
    std::ifstream lbl_file(label_path, std::ios::binary);

    if (!img_file.is_open() || !lbl_file.is_open()) {
        std::cerr << "Blad: Nie mozna otworzyc plikow MNIST!" << std::endl;
        return false;
    }

    int magic_number, n_rows, n_cols;

    img_file.read((char*)&magic_number, 4);
    img_file.read((char*)&num_images, 4);
    img_file.read((char*)&n_rows, 4);
    img_file.read((char*)&n_cols, 4);

    num_images = reverseInt(num_images); 
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);
    image_size = n_rows * n_cols;

    lbl_file.read((char*)&magic_number, 4);
    int num_labels;
    lbl_file.read((char*)&num_labels, 4);
    num_labels = reverseInt(num_labels);

    h_images.resize(num_images, std::vector<double>(image_size));
    h_labels.resize(num_images);

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            img_file.read((char*)&pixel, 1);
            h_images[i][j] = static_cast<double>(pixel) / 255.0; // Normalizacja 0-1
        }
        unsigned char label = 0;
        lbl_file.read((char*)&label, 1);
        h_labels[i] = static_cast<int>(label);
    }

    std::cout << "Wczytano " << num_images << " obrazow o rozmiarze " << n_rows << "x" << n_cols << std::endl;
    return true;

}

int FileReader::reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
