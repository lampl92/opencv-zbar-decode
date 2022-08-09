#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <sys/time.h>
#include <zbar.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
int threshold_value = 100;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;
int const sigma = 3;

const char* window_name = "Threshold Demo";
const char* trackbar_value = "Value";

using namespace std::chrono;
using namespace std;
using namespace cv;
using namespace zbar;

void getCamInfo(VideoCapture cam) {
    cout<<"CAP_PROP_FRAME_WIDTH " << cam.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout<<"CAP_PROP_FRAME_HEIGHT " << cam.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout<<"CAP_PROP_FPS " << cam.get(CAP_PROP_FPS) << endl;
    cout<<"CAP_PROP_EXPOSURE " << cam.get(CAP_PROP_EXPOSURE) << endl;
    cout<<"CAP_PROP_FORMAT " << cam.get(CAP_PROP_FORMAT) << endl; //deafult CV_8UC3?!
    cout<<"CAP_PROP_CONTRAST " << cam.get(CAP_PROP_CONTRAST) << endl;
    cout<<"CAP_PROP_BRIGHTNESS "<< cam.get(CAP_PROP_BRIGHTNESS) << endl;
    cout<<"CAP_PROP_SATURATION "<< cam.get(CAP_PROP_SATURATION) << endl;
    cout<<"CAP_PROP_HUE "<< cam.get(CAP_PROP_HUE) << endl;
    cout<<"CAP_PROP_POS_FRAMES "<< cam.get(CAP_PROP_POS_FRAMES) << endl;
    cout<<"CAP_PROP_FOURCC "<< cam.get(CAP_PROP_FOURCC) << endl;

    int ex = static_cast<int>(cam.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
    char EXT[] = {(char)(ex & 255) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    cout << "Input codec type: " << EXT << endl;

}

uint64_t timeSinceEpochMillisec() {
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
#define FRAME_RATE_LIMIT 5
int main(int argc, char **argv) {
    int cam_idx = 0;
    uint64_t prev_time, cur_time;
    char file_path[100];
    if (argc == 2) {
        cam_idx = atoi(argv[1]);
    }

    VideoCapture cap;
    int deviceID = cam_idx;
    int apiID = CAP_ANY;
    cap.open(deviceID, apiID);
    if (!cap.isOpened()) {
        cerr << "Could not open camera." << endl;
        exit(EXIT_FAILURE);
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FOURCC, CV_FOURCC('Y', 'U', 'Y', 'V'));
    getCamInfo(cap);
    
    // Create a zbar reader
    ImageScanner scanner;
    
    // Configure the reader
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
    int counter = 0;
    float scale = 2;
    prev_time = timeSinceEpochMillisec();
    for (;;) {
        // Capture an OpenCV frame
        Mat frame, frame_grayscale, frame_process;
	cur_time = timeSinceEpochMillisec();
	cap.read(frame);
	if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
	    continue;
        }
	if((cur_time - prev_time) < (1000 / FRAME_RATE_LIMIT)) {
	    continue;
	}
	prev_time = timeSinceEpochMillisec();
	//cout << timeSinceEpochMillisec()  << " : GET FRAME 1" << endl;
	counter++;
	resize(frame, frame, Size(), (float) 1/scale, (float) 1/ scale);
        // Convert to grayscale
        cvtColor(frame, frame_grayscale, COLOR_BGR2GRAY);
	//cout << timeSinceEpochMillisec() << " : 2" << endl;
	GaussianBlur(frame_grayscale, frame_grayscale, Size(5, 5), 0, 0);
	//cout << timeSinceEpochMillisec() << " : 3" << endl;
	adaptiveThreshold(frame_grayscale, frame_process, threshold_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 41 ,2 );
	//cout << timeSinceEpochMillisec() << " : 4" << endl;
        // Obtain image idata
        int width = frame_process.cols;
        int height = frame_process.rows;
        uchar *raw = (uchar *)(frame_process.data);

        // Wrap image data
        Image image(width, height, "Y800", raw, width * height);
        //cout << timeSinceEpochMillisec() << " : 5" << endl;
        // Scan the image for barcodes
        //int n = scanner.scan(image);
        if(scanner.scan(image) == 0) {
            //cout << timeSinceEpochMillisec() << " : FAILED" << endl;
	    sprintf(file_path, "/tmp/opencv_%d_fail.jpg", counter);
	}
        // Extract results
        for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
            time_t now;
            tm *current;
            now = time(0);
            current = localtime(&now);
            //cout << timeSinceEpochMillisec() << " : PASS" << endl;
	    sprintf(file_path, "/tmp/opencv_%d_pass.jpg", counter);

            // do something useful with results
            cout    << "[" << current->tm_hour << ":" << current->tm_min << ":" << setw(2) << setfill('0') << current->tm_sec << "] " << " "
                    << "decoded " << symbol->get_type_name()
                    << " symbol \"" << symbol->get_data() << '"' << endl;
	    line(frame_process, Point(symbol->get_location_x(0), symbol->get_location_y(0)), Point(symbol->get_location_x(1), symbol->get_location_y(1)), Scalar(0, 255, 0), 2, 8, 0);
            line(frame_process, Point(symbol->get_location_x(1), symbol->get_location_y(1)), Point(symbol->get_location_x(2), symbol->get_location_y(2)), Scalar(0, 255, 0), 2, 8, 0);
            line(frame_process, Point(symbol->get_location_x(2), symbol->get_location_y(2)), Point(symbol->get_location_x(3), symbol->get_location_y(3)), Scalar(0, 255, 0), 2, 8, 0);
            line(frame_process, Point(symbol->get_location_x(3), symbol->get_location_y(3)), Point(symbol->get_location_x(0), symbol->get_location_y(0)), Scalar(0, 255, 0), 2, 8, 0);

        }

        // Show captured frame, now with overlays!
        // clean up
	//imwrite(file_path, frame_process);
	cout << "duration: " << timeSinceEpochMillisec() - cur_time << " ms"<< endl;
        image.set_data(NULL, 0);
    }

    return 0;
}
