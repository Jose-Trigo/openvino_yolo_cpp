#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <numeric>
#include <vector>

static const float YOLO_CONFIDENCE = 0.2f;
static const float IOU_THRESHOLD   = 0.45f;

struct Detection {
    cv::Rect box;
    float score;
};

float IoU(const cv::Rect& a, const cv::Rect& b) {
    int interArea = (a & b).area();
    int unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? static_cast<float>(interArea) / unionArea : 0.0f;
}

std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh) {
    std::vector<int> indices;
    if (dets.empty()) return indices;

    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return dets[i].score > dets[j].score;
    });

    while (!order.empty()) {
        int idx = order.front();
        indices.push_back(idx);
        std::vector<int> rest;
        for (size_t k = 1; k < order.size(); ++k) {
            int j = order[k];
            if (IoU(dets[idx].box, dets[j].box) < iou_thresh) {
                rest.push_back(j);
            }
        }
        order.swap(rest);
    }
    return indices;
}

int main() {
    std::string video_path   = "brt_presentation.mp4";
    std::string model_dir    = "yolo_nano_v2_1_class_640_no_filter_openvino_model";
    std::string model_xml    = model_dir + "/yolo_nano_v2_1_class_640_no_filter.xml";
    std::string output_video = "output_cpp.mp4";

    std::cout << "OpenVINO YOLO Inference Benchmark (C++)\n";
    std::cout << "Model: " << model_xml << "\n";
    std::cout << "Video: " << video_path << "\n";

    // ---------------- OpenVINO: load model ----------------
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_xml);

    // ---------------- PrePostProcessor ----------------
    using namespace ov::preprocess;

    PrePostProcessor ppp(model);

    // Input description
    auto& input = ppp.input(0);

    input.tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ColorFormat::BGR);

    input.preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ColorFormat::RGB)
        .resize(ResizeAlgorithm::RESIZE_LINEAR)
        .scale({255.0f});

    input.model()
        .set_layout("NCHW");

    // Output description
    ppp.output(0)
        .tensor()
        .set_element_type(ov::element::f32);

    model = ppp.build();

    // ---------------- Compile model ----------------
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request   = compiled_model.create_infer_request();

    ov::Output<const ov::Node> output = compiled_model.output();

    // ---------------- Open video ----------------
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video.\n";
        return 1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps       = cap.get(cv::CAP_PROP_FPS);
    int width        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Total frames: " << total_frames << "\n";
    std::cout << "Resolution: " << width << "x" << height << ", FPS: " << fps << "\n";

    cv::VideoWriter writer(
        output_video,
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps > 0 ? fps : 25.0,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output file.\n";
        return 1;
    }

    std::vector<double> times;
    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        frame_count++;

        int img_h = frame.rows;
        int img_w = frame.cols;

        auto t0 = std::chrono::high_resolution_clock::now();

        // ---------------- Input tensor ----------------
        ov::Tensor input_tensor(
            ov::element::u8,
            ov::Shape{1, static_cast<size_t>(img_h), static_cast<size_t>(img_w), 3},
            frame.data
        );
        infer_request.set_input_tensor(input_tensor);

        // ---------------- Inference ----------------
        infer_request.infer();

        // ---------------- Get output ----------------
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        auto out_shape = output_tensor.get_shape(); // [1, 5, 8400]
        const float* out_data = output_tensor.data<const float>();

        int num_cols = static_cast<int>(out_shape[2]);
        std::vector<Detection> detections;
        detections.reserve(num_cols);

        for (int i = 0; i < num_cols; ++i) {
            float x     = out_data[0 * num_cols + i];
            float y     = out_data[1 * num_cols + i];
            float w     = out_data[2 * num_cols + i];
            float h     = out_data[3 * num_cols + i];
            float score = out_data[4 * num_cols + i];

            if (score < YOLO_CONFIDENCE) continue;

            float cx = x * img_w / 640.0f;
            float cy = y * img_h / 640.0f;
            float bw = w * img_w / 640.0f;
            float bh = h * img_h / 640.0f;

            int x1 = static_cast<int>(cx - bw / 2.0f);
            int y1 = static_cast<int>(cy - bh / 2.0f);
            int x2 = static_cast<int>(cx + bw / 2.0f);
            int y2 = static_cast<int>(cy + bh / 2.0f);

            detections.push_back({cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), score});
        }

        std::vector<Detection> final_dets;
        if (!detections.empty()) {
            std::vector<int> keep = nms(detections, IOU_THRESHOLD);
            for (int idx : keep) final_dets.push_back(detections[idx]);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(dt);

        // ---------------- Draw + write ----------------
        cv::Mat annotated = frame.clone();
        for (const auto& det : final_dets) {
            cv::rectangle(annotated, det.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(
                annotated,
                cv::format("%.2f", det.score),
                det.box.tl() + cv::Point(0, -5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 255, 0),
                2
            );
        }

        writer.write(annotated);

        if (frame_count % 100 == 0) {
            std::cout << "Processed " << frame_count << "/" << total_frames << " frames\n";
        }
    }

    cap.release();
    writer.release();

    if (!times.empty()) {
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::cout << "\n==================================================\n";
        std::cout << "Frames processed: " << frame_count << "\n";
        std::cout << "Average total time: " << avg * 1000.0 << " ms\n";
        std::cout << "FPS (end-to-end): " << 1.0 / avg << "\n";
        std::cout << "Annotated video saved as: " << output_video << "\n";
        std::cout << "==================================================\n";
    }

    return 0;
}
