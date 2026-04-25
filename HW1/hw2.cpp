#include <opencv2/opencv.hpp>
#include <string>
#include <print>
#include <numbers>

void drawImage(const cv::Affine3d& P, const cv::Matx33d& cameraMatrix, const cv::Mat& wall, bool draw=false)
{
    static cv::Mat image = cv::Mat::zeros(640, 640, CV_8UC3);

    const int imHeight = 640, imWidth = 640;

    for (auto i{0uz}; i < imHeight; i++)
    {
        for (auto j{0uz}; j < imWidth; j++)
        {
            cv::Matx33d R = P.rotation();
            cv::Vec3d t = P.translation();

            cv::Vec3d pixelsProjective{static_cast<double>(i), static_cast<double>(j), 1.};
            cv::Vec3d X_camera = cameraMatrix.inv() * pixelsProjective;
            double k = ((R.inv() * t)[2] + 1000.) / (R.inv() * X_camera)[2];
            cv::Vec3d X_world = R.inv() * (k * X_camera - t);
            cv::Vec2d pixWall{X_world[0] + wall.rows / 2, X_world[1] + wall.cols / 2};

            if (pixWall[0] >= 0. && pixWall[0] < wall.rows && pixWall[1] >= 0. && pixWall[1] < wall.cols)
            {
                image.at<cv::Vec3b>(i, j) = wall.at<cv::Vec3b>(static_cast<int>(pixWall[0]), static_cast<int>(pixWall[1]));
            }
        }
    }
    
    if (draw)
    {
        cv::imshow("Room", image);
        cv::waitKey(0);
    }
}


int main(int argc, char *argv[])
{
    std::string path{"./pibble.jpg"};
    cv::Mat wall1{cv::imread(path)};

    //std::println("Dimensions of wall1: {}, {}", wall1.rows, wall1.cols);

    double fx = 300., fy = 300., cx = 320., cy = 320.;
    cv::Matx33d cameraMatrix{fx, 0., cx, 0., fy, cy, 0., 0., 1.};

    cv::Vec3d rvec1{0., 0., 0.}, t1{0., 0., 0.};
    cv::Affine3d P1{rvec1, t1};
    drawImage(P1, cameraMatrix, wall1);
    

    cv::Mat red{cv::Mat::ones(611, 734, CV_8UC3)};
    red.setTo(cv::Scalar(0, 0, 255));
    cv::Mat wall2{cv::Mat::zeros(611, 1400, CV_8UC3)};
    cv::resize(red, wall2, wall2.size(), 0, 0);

    cv::Vec3d rvec2{std::numbers::pi / 3, 0., 0.}, t2{0., 1500., 1000.};
    cv::Affine3d P2{rvec2, t2};
    drawImage(P2, cameraMatrix, wall2, true);

    return 0;
}
