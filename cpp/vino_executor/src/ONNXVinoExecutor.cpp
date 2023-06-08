#include <vino_executor/ONNXVinoExecutor.h>

namespace {
constexpr int expected_image_type = CV_32FC3;
constexpr ov::element::Type input_type = ov::element::f32; // the input type for all input tensors

ov::Tensor hasNoMasksTensor() {
    ov::Tensor result(input_type, ov::Shape{ 1 });
    auto data = result.data<float>();
    data[0] = 0.f;

    return result;
}

ov::Tensor dummyMaskTensor() {
    ov::Tensor result(input_type, ov::Shape{ 1, 1, 256, 256 });
    auto data = result.data<float>();
    for (size_t i = 0, size = result.get_size(); i < size; ++i)
        data[i] = 0.f;

    return result;
}

ov::Tensor points2Tensor(const std::vector<cv::Point2f>& points) {
    auto result = ov::Tensor(input_type, ov::Shape{ 1, points.size(), 2 });
    std::memcpy(result.data<float>(), points.data(), points.size() * sizeof(cv::Point2f));

    return result;
}

ov::Tensor labels2Tensor(const std::vector<float>& labels) {
    auto result = ov::Tensor(input_type, ov::Shape{ 1, labels.size() });
    std::memcpy(result.data<float>(), labels.data(), labels.size() * sizeof(float));

    return result;
}
}

namespace vino_executor {

ONNXVinoExecutor::ONNXVinoExecutor(ov::Core core,
                                   std::filesystem::path im_encoder_path,
                                   std::filesystem::path the_rest_path)
    : m_core(core)
{
    std::cout << "Loading image encoder from: " << im_encoder_path.string() << "\t...";
    std::shared_ptr<ov::Model> im_enc = core.read_model(im_encoder_path);
    std::cout << "\tsuccessful" << std::endl;

    std::cout << "Setting up image encoder preprocessor" << "\t...";
    ov::preprocess::PrePostProcessor ppp(im_enc);
    ppp.input().tensor().set_element_type(input_type).set_layout("NHWC"); // supposing the input image has this layout
    ppp.input().model().set_layout("NCHW"); // expect the image encoder to have this layout

    m_im_encoder = ppp.build();
    std::cout << "\tsuccessful" << std::endl;

    std::cout << "Compiling the image encoder to a device" << "\t...";
    m_im_enc_compiled = m_core.compile_model(m_im_encoder, "CPU");
    std::cout << "\tsuccessful" << std::endl;

    std::cout << "Loading the model tail from: " << the_rest_path.string() << "\t...";
    m_the_rest = core.read_model(the_rest_path);
    std::cout << "\tsuccessful" << std::endl;

    std::cout << "Compiling the tail to a device" << "\t...";
    m_the_rest_compiled = m_core.compile_model(m_the_rest, "CPU");
    std::cout << "\tsuccessful" << std::endl;

    std::cout << "Creating infer requests" << "\t...";
    m_im_enc_infer = m_im_enc_compiled.create_infer_request();
    m_the_rest_infer = m_the_rest_compiled.create_infer_request();
    std::cout << "\tsuccessful" << std::endl;
}

ONNXVinoExecutor::~ONNXVinoExecutor() = default;

cv::Size ONNXVinoExecutor::input_size() const {
    const auto& shape = m_im_encoder->input().get_shape();
    return cv::Size(static_cast<int>(shape[2]), static_cast<int>(shape[1])); // N[HW]C --> cv::Size(W, H)
}

cv::Size ONNXVinoExecutor::output_size() const {
    const auto& shape = m_the_rest->output(0).get_shape();
    return cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2])); // NC[HW] --> cv::Size(W, H)
}

bool ONNXVinoExecutor::encode_image(cv::Mat input_image) {
    if (input_image.type() != expected_image_type)
        throw std::runtime_error("ONNXVinoExecutor::encode_image: Expected / actual input image type mismatch (must be CV_32FC3");

    if (input_image.size() != input_size()) {
        std::ostringstream ss;
        ss << "ONNXVinoExecutor::encode_image: Input image must be of size " << input_size() << std::endl;
        throw std::runtime_error(ss.str());
    }

    // wrapping image data by ov::Tensor without memory allocation
    // we also assume that the image data is in contiguous memory
    ov::Tensor input_tensor = ov::Tensor(m_im_encoder->input().get_element_type(), 
                                         m_im_encoder->input().get_shape(),
                                         input_image.data);

    m_im_enc_infer.set_input_tensor(input_tensor);
    m_im_enc_infer.infer();

    return true;
}

cv::Mat ONNXVinoExecutor::predict(const std::vector<cv::Point2f>& points, const std::vector<float>& labels) {
    if (points.empty()) throw std::runtime_error("ONNXVinoExecutor::predict: empty inputs");

    if (points.size() != labels.size())
        throw std::runtime_error("ONNXVinoExecutor::predict: points / labels size mismatch");

    static const ov::Tensor has_no_masks = hasNoMasksTensor();
    static const ov::Tensor dummy_mask = dummyMaskTensor(); // although we don't use any masks, this tensor must be explicitly nullified

    // filling the inputs - their names and shapes should coinside with the ones specified during the export to ONNX-format
    m_the_rest_infer.set_tensor("image_embeddings", m_im_enc_infer.get_output_tensor());
    m_the_rest_infer.set_tensor("point_coords", points2Tensor(points));
    m_the_rest_infer.set_tensor("point_labels", labels2Tensor(labels));
    m_the_rest_infer.set_tensor("mask_input", dummy_mask);
    m_the_rest_infer.set_tensor("has_mask_input", has_no_masks);

    m_the_rest_infer.infer();

    // Getting the result and transforming it into cv::Mat
    const ov::Tensor& output_tensor = m_the_rest_infer.get_output_tensor(0);
    return cv::Mat(output_size(), CV_32FC1, output_tensor.data<float>()).clone();
}

}
