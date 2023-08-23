#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <openvino/openvino.hpp>

#include <cppsam/SAMModel.h>
#include <vino_executor/ONNXVinoExecutor.h>

#include <cli11/CLI11.hpp>

using namespace std::filesystem;
using namespace std::chrono;

namespace {

cv::Mat applyMask(const cv::Mat& image, const cv::Mat& mask) {
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());
    image.copyTo(result, mask);

    return result;
}

}

int main(int argc, char** argv) {
    CLI::App app{"An example of running Segment-Anything with OpenCV and OpenVINO in C++"};

    std::string export_path;
    app.add_option("-x,--export_path",
                   export_path,
                   "A path to ONNX-exported segment-anything model"
    )->required()->type_name("path")->check(CLI::ExistingDirectory);
    std::cout << export_path << std::endl;

    std::string image_path;
    app.add_option("-i,--image_path",
                   image_path,
                   "A path to an RGB image to process"
    )->required()->type_name("path")->check(CLI::ExistingFile);
    std::cout << image_path << std::endl;

    std::vector<float> input_coordinates;
    app.add_option("-p,--input_points",
                   input_coordinates,
                   "Coordinates of sparse input prompts, corresponding to both point inputs and box inputs,\n"
                   "e.g. 926, 926, 806, 918, 0, 0.\n"
                   "Boxes are encoded using two points, one for\n"
                   "the top-left corner and one for the bottom-right corner.\n"
                   "Each pair of adjacent comma-separated values is treated as a 2d-point.\n"
                   "Note that input_points require additional padding point 0, 0\n"
                   "in case there is no box input for the model"
    )->required()->type_name("vector<float>")->delimiter(',');

    std::vector<float> input_labels;
    app.add_option("-l,--input_labels",
                   input_labels,
                   "Labels for the sparse input prompts, e.g. 1, 0, -1\n"
                   "\t0 is a negative input point,\n"
                   "\t1 is a positive input point,\n"
                   "\t2 is a top-left box corner,\n"
                   "\t3 is a bottom-right box corner,\n"
                   "\t-1 is a padding point.\n"
                   "The number of labels must coincide with the number of input points\n"
                   "Note that input_labels require additional padding label -1\n"
                   "in case there is no box input for the model"
    )->required()->type_name("vector<float>")->delimiter(',')->check(CLI::Range(-1.f, 3.f));

    std::string hardware;
    app.add_option("-w,--hardware",
                   hardware,
                   "The hardware to perform inference: CPU (default), GPU or MYRIAD"
    )->default_str("CPU")->type_name("Enum")->transform(CLI::IsMember({"CPU", "GPU", "MYRIAD"}));

    std::cout << hardware << std::endl;

    CLI11_PARSE(app, argc, argv);

    try {
        const std::filesystem::path im_enc_path = std::filesystem::path(export_path).append("image_encoder.onnx");
        const std::filesystem::path the_rest_path = std::filesystem::path(export_path).append("the_rest.onnx");

        if (!exists(im_enc_path)) {
            std::stringstream ss;
            ss << "Path " << im_enc_path.string() << " doesn't exist" << std::endl;
            throw std::runtime_error(ss.str());
        }
        if (!exists(the_rest_path)) {
            std::stringstream ss;
            ss << "Path " << the_rest_path.string() << " doesn't exist" << std::endl;
            throw std::runtime_error(ss.str());
        }

        if (input_coordinates.size() % 2 > 0)
            throw std::runtime_error("The vector of input points must have even size");

        std::vector<cv::Point2f> input_points;
        for (size_t i = 0; i < input_coordinates.size(); i = i + 2)
            input_points.emplace_back(input_coordinates[i], input_coordinates[i + 1]);

        if (input_points.size() != input_labels.size())
            throw std::runtime_error("The number of points and labels must coincide");

        auto test_im = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::cvtColor(test_im, test_im, cv::COLOR_BGR2RGB); // SAM awaits an image in RGB format

        // Setting up and running the model
        cppsam::SAMModel model(std::make_shared<vino_executor::ONNXVinoExecutor>(ov::Core(), im_enc_path, the_rest_path));

        // Making inference
        auto t0 = high_resolution_clock::now();
        model.setInput(test_im);
        cv::Mat result = model.predict(input_points, input_labels);
        auto t1 = high_resolution_clock::now();

        std::cout << "Inference in " << duration_cast<milliseconds>(t1 - t0).count() << " ms" << std::endl;

        //preparing for showing the results
        cv::resize(result, result, cv::Size(), 0.4, 0.4); // resize for convenient representation
        cv::resize(test_im, test_im, cv::Size(), 0.4, 0.4); // resize for convenient representation
        cv::cvtColor(test_im, test_im, cv::COLOR_RGB2BGR); // converting back into BGR format for presenting
        cv::Mat masked = applyMask(test_im, result);

        cv::imshow("Result image", masked);
        cv::imshow("Original image", test_im);
        cv::waitKey(0);
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cout << "Error in main: unknown error";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
