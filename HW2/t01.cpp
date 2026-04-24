#include <opencv2/opencv.hpp>
#include <print>
#include <iostream>
#include <vector>
#include "hw2utils.hpp"


int main(int argc, char* argv[]) {
    const cv::Vec3d rvec{0.1, -0.2, 0.3}, t{1.1, 2.0, 3.};
    const cv::Affine3d affineTransform{rvec, t};
    constexpr double fx{300.}, fy{300.}, cx{320.}, cy{320.};
    const cv::Matx33d K{fx, 0., cx, 0., fy, cy, 0., 0., 1.};

	cv::Mat Rt1{};
	cv::hconcat(K, cv::Mat::zeros(3, 1, CV_64F), Rt1);
	const cv::Mat P1{Rt1};

	cv::Mat Rt2{cv::Mat{affineTransform.matrix}.rowRange(0, 3)};
	const cv::Mat P2{K * Rt2};
	

    const std::vector<cv::Point3d> xWorldPoints{
        {3, 2, 5},
        {10, -3, 5.5},
        {-3, 1, 4.5},
        {8, 2, 4},
        {-2, -3, 4},
        {8, 3, 10},
        {-5, 4, 6},
        {2, -6, 7}
    };

	std::vector<cv::Point2d> pts1{hw2::projectPoints(xWorldPoints, P1)}, pts2{hw2::projectPoints(xWorldPoints, P2)};

	auto [R_ret, t_ret]{hw2::findPose(pts1, pts2, K)};

	cv::Mat rvec_ret{};
	cv::Rodrigues(R_ret, rvec_ret);

	std::println("Ground truth:");
	
	std::cout << "Rotation vector\n";
	std::cout << rvec << "\n";

	std::cout << "Rotation matrix\n";
	std::cout << cv::Mat{affineTransform.matrix}.rowRange(0, 3) << "\n";

	std::cout << "Translation vector:\n";
	std::cout << t << "\n";

	std::println("");

	std::println("Found using cv::findEssentialMat()");
	
	std::cout << "Rotation vector\n";
	std::cout << rvec_ret << "\n";

	std::cout << "Translation direction:\n";
	std::cout << t_ret << "\n";

	std::println("");

	std::println("Found using cv::solvePnP()");

	auto [rodr_pnp, t_pnp]{hw2::findPosePnP(xWorldPoints, pts2, K)};

	std::cout << "Rotation vector\n";
	std::cout << rodr_pnp << "\n";

	std::cout << "Translation direction:\n";
	std::cout << t_pnp << "\n";	

}
