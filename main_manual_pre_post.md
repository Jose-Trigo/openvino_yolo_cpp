#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <numeric>
#include <vector>
#include <cstring>

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
    std::string output_video = "output_cpp.avi";  // use AVI + MJPG

    std::cout << "OpenVINO YOLO Inference Benchmark (C++)\n";
    std::cout << "Model: " << model_xml << "\n";
    std::cout << "Video: " << video_path << "\n";

    // ---------------- OpenVINO: load model ----------------
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_xml);

    auto input_port = model->input();
    std::cout << "Model input shape: ";
    for (auto d : input_port.get_shape()) std::cout << d << " ";
    std::cout << "\n";

    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request   = compiled_model.create_infer_request();

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

    // Use MJPG in AVI – very robust
    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');

    cv::VideoWriter writer(
        output_video,
        fourcc,
        fps > 0 ? fps : 25.0,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output file for writing.\n";
        return 1;
    }

    std::vector<double> times;
    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cout << "No more frames or failed to read frame.\n";
            break;
        }
        frame_count++;

        auto t0 = std::chrono::high_resolution_clock::now();

        // -------- Manual preprocessing: BGR -> RGB, resize to 640x640, normalize --------
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 640));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0f / 255.0f);

        // HWC -> CHW
        std::vector<cv::Mat> chw(3);
        cv::split(float_img, chw); // 3 x (640x640)

        ov::Tensor input_tensor(ov::element::f32, {1, 3, 640, 640});
        float* data = input_tensor.data<float>();

        for (int c = 0; c < 3; ++c) {
            std::memcpy(
                data + c * 640 * 640,
                chw[c].data,
                640 * 640 * sizeof(float)
            );
        }

        infer_request.set_input_tensor(input_tensor);

        // ---------------- Inference ----------------
        infer_request.infer();

        // ---------------- Get output ----------------
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        auto out_shape = output_tensor.get_shape(); // expect [1, 5, 8400]
        const float* out_data = output_tensor.data<const float>();

        if (frame_count == 1) {
            std::cout << "Output shape: ";
            for (auto d : out_shape) std::cout << d << " ";
            std::cout << "\n";
        }

        if (out_shape.size() != 3 || out_shape[1] != 5) {
            std::cerr << "Unexpected output shape, skipping frame.\n";
            continue;
        }

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

            float cx = x * width / 640.0f;
            float cy = y * height / 640.0f;
            float bw = w * width / 640.0f;
            float bh = h * height / 640.0f;

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

        if (frame_count % 50 == 0) {
            std::cout << "Processed " << frame_count << "/" << total_frames << "\n";
        }
    }

    cap.release();
    writer.release();

    std::cout << "Total frames actually processed: " << frame_count << "\n";

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
