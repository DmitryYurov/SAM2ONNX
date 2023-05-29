#include <openvino/openvino.hpp>

#include <cppsam/SAMModel.h>
#include <vino_executor/ONNXVinoExecutor.h>

int main() {
    auto core = std::make_shared<ov::Core>();
    try {
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cout << "Error in main: unknown error";
        return -1;
    }

    return 0;
}
