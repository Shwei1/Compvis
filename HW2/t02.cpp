#include "hw2utils.hpp"
#include <print>
#include <iostream>

int main(int argc, char* argv[]) {
    const cv::Vec3d rvec{0.1, -0.2, 0.3}, t{1.1, 2.0, 3.};
    const cv::Affine3d affineTransform{rvec, t};
    const double fx{300.}, fy{300.}, cx{320.}, cy{320.};
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

	//auto [R_ret, t_ret]{findPose(pts1, pts2, K)};
	auto [rodr_pnp, t_rodr]{hw2::findPosePnP(xWorldPoints, pts2, K)};
	cv::Mat R_pnp{};
	cv::Rodrigues(rodr_pnp, R_pnp);

	std::cout << "True rotation matrix:\n";
	std::cout << cv::Mat{affineTransform.matrix}.rowRange(0, 3).colRange(0, 3) << "\n";

	std::cout << "Reconstructed with PnP:\n";
	std::cout << R_pnp << "\n";


	std::cout << "Gram-Schmidt normalised:\n";
	std::cout << hw2::GramSchmidtProcess(R_pnp) << "\n";

	return 0;
}
