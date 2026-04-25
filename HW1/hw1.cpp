#include <opencv2/opencv.hpp>
#include <print>

void drawImage(const cv::Affine3d& P, const cv::Matx33d& cameraMatrix, const cv::Mat& wall, bool draw=false)
{
    cv::Mat image = cv::Mat::zeros(640, 640, CV_8UC3);

    const int imHeight = 640, imWidth = 640;

    for (auto i{0uz}; i < imHeight; i++)
    {
        for (auto j{0uz}; j < imWidth; j++)
        {
            cv::Matx33d R = P.rotation();
            cv::Vec3d t = P.translation();

            cv::Vec3d pixelsProjective{static_cast<double>(i), static_cast<double>(j), 1.};
            cv::Vec3d X_camera = cameraMatrix.inv() * pixelsProjective;
            double k = ((R.inv() * t)[2] + 2500.) / (R.inv() * X_camera)[2];
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
        cv::waitKey(1);
    }
}


int main(int argc, char *argv[])
{
    std::string path{"./pibble.jpg"};
    //cv::Mat wall1{cv::imread(path)};
//
    //std::println("Dimensions of wall1: {}, {}", wall1.rows, wall1.cols);
//
    //cv::Vec3d rvec1{0., 0., 0.}, t1{0., 0., 0.};
    //cv::Affine3d P1{rvec1, t1};
//
    double fx = 300., fy = 300., cx = 320., cy = 320.;
    cv::Matx33d cameraMatrix{fx, 0., cx, 0., fy, cy, 0., 0., 1.};


    //drawImage(P1, cameraMatrix, wall1);

    //cv::Mat wall2{cv::Mat::zeros(611, 1000, CV_8UC3)};
    //cv::resize(wall1, wall2, wall2.size(), 0, 0);
//
    //cv::Vec3d rvec2{0.15, 0., 0.5}, t2{0., -736., 0.};
    //cv::Affine3d P2{rvec2, t2};
    //drawImage(P2, cameraMatrix, wall2, true);

    cv::Mat dog{cv::imread(path)};

    ///* Walking towards */
    //for (auto s{1000.}; s < 2000.; s -= 50.)
    //{
        //std::println("{}", s);
        //cv::Vec3d rvec3{0., 0., 0.}, t3{0., 0., s};
        //cv::Affine3d P3{rvec3, t3};
        //drawImage(P3, cameraMatrix, dog, true);
    //}
//
    ///* Rotating 1 */
    //for (auto s{0.}; s < 6.28; s += 0.05)
    //{
        //std::println("{}", s);
        //cv::Vec3d rvec3{s, 0., 0.}, t3{0., 0., 0.};
        //cv::Affine3d P3{rvec3, t3};
        //drawImage(P3, cameraMatrix, dog, true);
    //}
//
    ///* Rotating 2 */
    //for (auto s{0.}; s < 6.28; s += 0.05)
    //{
        //std::println("{}", s);
        //cv::Vec3d rvec3{0., s, 0.}, t3{0., 0., 0.};
        //cv::Affine3d P3{rvec3, t3};
        //drawImage(P3, cameraMatrix, dog, true);
    //}
    //
    
    /* Rotating 3 */
    for (auto s{0.}; s < 6.28; s += 0.05)
    {
        std::println("{}", s);
        cv::Vec3d rvec3{0., 0., s}, t3{0., 0., 0.};
        cv::Affine3d P3{rvec3, t3};
        drawImage(P3, cameraMatrix, dog, true);
    }

    
    return 0;
}
