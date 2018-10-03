#include<iostream>
#include <string.h>

#include"nnpack/layers.hpp"

// options
struct options {
    string data_path;
    string model_para_path;
};

static struct options parse_options(int argc, char** argv) {
    struct options options = {
        .data_path = "None",
    };
    for (int argi = 1; argi < argc; argi += 1) {
        if (strcmp(argv[argi], "--input-data") == 0) {
            if (strcmp(argv[argi + 1], "") == 0) {
                std::cerr << "Error: please assign the data path" << std::endl;
                exit(1);
            }
            options.data_path = argv[argi + 1];
            argi += 1;
        }
        else if (strcmp(argv[argi], "--model-para") == 0) {
            if (strcmp(argv[argi + 1], "") == 0) {
                std::cerr << "Error: please assign the model para dir path" << std::endl;
                exit(1);
            }
            options.model_para_path = argv[argi + 1];
            argi += 1;
        }
        else {
            std::cerr << "Error: argu not recognized" << std::endl;
            exit(1);
        }
    }
    return options;
};

void BuildModel(Net& model, const struct options& options ) {
    // data layer
    DataLayer* data_layer = new DataLayer(options.data_path, 3, 224, "data");
    model.addLayer(data_layer);
    
    // first conv layer
    ConvLayer* conv1 = new ConvLayer(data_layer->_output,
            3, 64, 224, 3, 7, 2, true, "conv1");
    model.addLayer(conv1);

    model.initPara(options.model_para_path);

    model.prepareComputation();
    return;
}

int main(int argc, char* argv[]) {
    std::cout << "\n============== Test ResNet ==============\n" << std::endl;
    const struct options options = parse_options(argc, argv);

    std::cout << "--> Input Data Path: " << options.data_path << std::endl;
    std::cout << "--> Model Para Dir Path: " << options.model_para_path << std::endl;
    Net model;

    BuildModel(model, options);

    // fetch input data
    model.fetchData();
    model.forward();
    return 0;
}
