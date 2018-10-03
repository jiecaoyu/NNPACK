#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include <nnpack.h>

using namespace std; 

#define DataType float


/*
 * *********************
 * Original Layer class
 * *********************
 */
class Layer {
    public:
    string _type;
    string _name;
    pthreadpool_t _threadpool = NULL;
    virtual void forward() {};
    virtual void fetchData() {
        std::cerr << "Error: DataLayer need to be the first layer" << std::endl;
        exit(1);
        return;
    }

    virtual void initPara(const string para_path) {
        std::cerr << "Error: wrong initPara" << std::endl;
        exit(1);
        return;
    };

    virtual DataType* getOutput() {
        std::cerr << "Error: wrong getOutput" << std::endl;
        exit(1);
        return NULL;
    };
};

/*
 * *********************
 * Convolutional Layer
 * *********************
 */

class ConvLayer: public Layer {
    public:
    DataType* _input;
    DataType* _output;
    DataType* _kernel;
    DataType* _bias;
    int _input_channels;
    int _output_channels;
    struct nnp_padding _input_padding;
    struct nnp_size _input_size;
    struct nnp_size _output_size;
    struct nnp_size _kernel_size;
    struct nnp_size _stride;
    bool _relu;
    ConvLayer(DataType* input,
            const size_t input_channels,
            const size_t output_channels,
            const size_t input_size,
            const size_t input_padding,
            const size_t kernel_size,
            const size_t stride,
            const bool relu,
            const string name) {
        std::cout << "==> Build Layer [" << name << "]\n";
        // configure
        _type = "Conv";
        _input_channels = input_channels;
        _output_channels = output_channels;
        _input_size = (struct nnp_size) {input_size, input_size};
        _input_padding = (struct nnp_padding) {input_padding, input_padding,
            input_padding, input_padding };
        _kernel_size = (struct nnp_size) {kernel_size, kernel_size};
        _stride = (struct nnp_size) {stride, stride};
        _relu = relu;
        _name = name;

        _input = input;

        _output_size = (struct nnp_size) {
            (_input_size.height - _kernel_size.height + 2 * input_padding) / _stride.height + 1,
            (_input_size.width - _kernel_size.width + 2 * input_padding) / _stride.width + 1
        };

        // allocate kernel and bias
        _kernel = new DataType[_input_channels * _output_channels
            * _kernel_size.height * _kernel_size.width];
        _bias = new DataType[_output_channels];

        // allocate output
        _output = new DataType[output_channels
            * _output_size.height * _output_size.width];
    }
    virtual void forward() {
        std::cout << "==> Layer " << _name << " forward" << std::endl;
        enum nnp_status status = nnp_status_success;
        status = nnp_convolution_inference(
                nnp_convolution_algorithm_implicit_gemm,
                nnp_convolution_transform_strategy_compute,
                _input_channels, _output_channels,
                _input_size, _input_padding, _kernel_size, _stride,
                _input, _kernel, _bias, _output,
                NULL, NULL,
                _relu? nnp_activation_relu : nnp_activation_identity,
                NULL,
                _threadpool, NULL);
        if (status != nnp_status_success) {
            std::cerr << "Error: compuation of layer ["
                << _name << "] failed" << std::endl;
            exit(1);
        }
    };

    virtual void initPara(const string para_path) {
        std::cout << "==> Initializing Layer [" << _name << "] with data "\
            << para_path <<"\n";
        ifstream para_file (para_path, ios::in | ios::binary);
        if (para_file.is_open()) {
            para_file.read((char*)_kernel, _input_channels * _output_channels
                    * _kernel_size.height * _kernel_size.width * sizeof(DataType));
            para_file.read((char*)_bias, _output_channels * sizeof(DataType));
            para_file.close();
        }
        else {
            std::cerr << "Error: para_file not exist" << std::endl;
            exit(1);
        }
        return;
    };

    virtual DataType* getOutput() {
        return _output;
    }

    ~ConvLayer() {
        delete[] _kernel;
        delete[] _bias;
        delete[] _output;
    }
};

/*
 * *********************
 * Data Layer
 * *********************
 */

class DataLayer: public Layer {
    string _input_path;
    const int _data_dim = 3;
    public:
    DataType* _output;
    size_t _input_channels;
    struct nnp_size _input_size;
    DataLayer(const string data_path,
            const size_t input_channels,
            const size_t input_size,
            const string name) {
        std::cout << "==> Build Layer [" << name << "]\n";
        _type = "Data";
        _input_path = data_path;
        _input_channels = input_channels;
        _input_size = (struct nnp_size) {input_size, input_size};
        _output = new DataType[_input_channels
            * _input_size.height * _input_size.width];
        _name = name;
    }
    virtual void forward() {
        std::cout << "==> Layer " << _name << " forward" << std::endl;
    };

    virtual void fetchData() {
        std::cout << "==> Fetch the input data" << std::endl;
        ifstream data_file (_input_path, ios::in | ios::binary);
        if (data_file.is_open()) {
            int32_t img_size[_data_dim];
            data_file.read((char*)img_size,  _data_dim * sizeof(int32_t));
            fprintf(stdout, "--> Input size: %d %d %d\n",
                    img_size[0], img_size[1], img_size[2]);

            if ((img_size[0] != _input_channels)
                    || (img_size[1] != _input_size.height)
                    || (img_size[2] != _input_size.width)) {
                std::cerr << "Error: input image has a wrong size" << std::endl;
                exit(1);
            }
            const int input_data_size = img_size[0] * img_size[1] * img_size[2];
            data_file.read((char*)_output,  input_data_size * sizeof(DataType));
            data_file.close();
        }
        else {
            std::cerr << "Error: data_file not exist" << std::endl;
            exit(1);
        }
        return;
    }

    ~DataLayer() {
        delete[] _output;
    }
};


/*
 * *********************
 * Network
 * *********************
 */

class Net {
    public:
        vector<Layer*> layers;
        pthreadpool_t _threadpool = NULL;
        Net() {
        }

        void addLayer(Layer* layer) {
            layers.push_back(layer);
            return;
        };

        void forward() {
            std::cout << "==> Forwarding Model\n";
            for (Layer* layer: layers) {
                layer->forward();
            }
            return;
        };

        void initPara(const string model_para_path) {
            std::cout << "==> Initializing Model\n";
            for (Layer* layer: layers) {
                if (layer->_type == "Conv")
                    layer->initPara(model_para_path + "/" + layer->_name);
            }
            return;
        };

        void prepareComputation() {
            enum nnp_status init_status = nnp_initialize();
            if (init_status != nnp_status_success) {
                fprintf(stderr, "NNPACK initialization failed: error code %d\n", init_status);
                exit(EXIT_FAILURE);
            }
            std::cout << "==> Prepare the Computation\n";
            _threadpool = pthreadpool_create(0);
            printf("==> Assign threadpool with %zu threads\n", pthreadpool_get_threads_count(_threadpool));
            for (Layer* layer: layers) {
                layer->_threadpool = _threadpool;
            }
            return;
        }

        void fetchData() {
            layers[0]->fetchData();
        }
};
