#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <openvino/openvino.hpp>

#include <cppsam/SAMModel.h>
#include <vino_executor/ONNXVinoExecutor.h>

using namespace std::filesystem;

int main() {
    try {
        const std::filesystem::path im_enc_path = "C:\\Work\\Develop\\SAM2ONNX\\export\\image_encoder.onnx";
        const std::filesystem::path the_rest_path = "C:\\Work\\Develop\\SAM2ONNX\\export\\the_rest.onnx";

        if (!exists(im_enc_path) || !exists(the_rest_path))
            return EXIT_FAILURE;

        auto test_im = cv::imread("C:\\Work\\Develop\\SAM2ONNX\\data\\test_image.jpg", cv::IMREAD_COLOR);
        cv::cvtColor(test_im, test_im, cv::COLOR_BGR2RGB);

        // Setting up and running the model
        cppsam::SAMModel model(std::make_shared<vino_executor::ONNXVinoExecutor>(ov::Core(), im_enc_path, the_rest_path));
        model.setInput(test_im);
        cv::Mat result = model.predict(std::vector{ cv::Point2f(926, 926), cv::Point2f(806, 918), cv::Point2f(0, 0) }, std::vector{ 1.f, 0.f, -1.f });
        cv::imshow("Result mask", result);
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
