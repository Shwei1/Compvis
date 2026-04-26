#include <opencv2/opencv.hpp>
#include <ranges>
#include <iostream>
#include <random>
#include <csignal>
#include "hw3utils.hpp"

[[noreturn]] void handle(int signal) {
	std::cerr << "\033[1;31mCalculation failed.\033[0m\n";
	std::exit(1);
}


int main(int argc, char* argv[]) {
	std::signal(SIGABRT, handle);

    const cv::Vec3d rvec{0.1, -0.2, 0.3}, t{1.1, 2.0, 3.};
    const cv::Affine3d affineTransform{rvec, t};
    const double fx{300.}, fy{300.}, cx{320.}, cy{320.};
    const cv::Matx33d K{fx, 0., cx, 0., fy, cy, 0., 0., 1.};

	const cv::Mat Rt{cv::Mat{affineTransform.matrix}.rowRange(0, 3)};
	const cv::Mat P{K * Rt};

	const cv::Matx33d R{affineTransform.rotation()};
	const cv::Vec3d t_{affineTransform.translation()};

	std::random_device rd{};
	std::mt19937 gen{rd()}; // rd()
	std::normal_distribution<double> noise(0., 1.);
	std::uniform_real_distribution<double> prob(0., 1.);
	std::uniform_real_distribution<double> outlier(-100., 100.);


	const std::vector<cv::Point3d> scenePoints = {
		{0.0, 0.0, 5.0},
		{0.5, 0.0, 5.0},
		{-0.5, 0.0, 5.0},
		{0.0, 0.5, 5.0},
		{0.0, -0.5, 5.0},

		{1.0, 1.0, 7.0},
		{-1.0, 1.0, 7.0},
		{1.0, -1.0, 7.0},
		{-1.0, -1.0, 7.0},

		{2.0, 0.0, 6.0},
		{-2.0, 0.0, 6.0},
		{0.0, 2.0, 6.0},
		{0.0, -2.0, 6.0},

		{1.5, -1.5, 8.0},
		{-1.5, 1.5, 8.0},
		{2.0, 2.0, 9.0},
		{-2.0, -2.0, 9.0},

		{0.3, -0.3, 3.0},
		{-0.3, 0.3, 3.0},
		{0.0, 0.0, 4.0},

		{3.0, 1.0, 7.0},
		{-3.0, -1.0, 7.0},
		{1.0, 3.0, 8.0},
		{-1.0, -3.0, 8.0},
		{2.5, -2.5, 10.0},
	};


	auto ptsTrue{hw3::projectPoints(scenePoints, P)};

	std::vector<hw3::Correspondence> correspondencesTrue{};

	for (auto&& [worldPoint, imagePoint] : std::views::zip(scenePoints, ptsTrue)) {
		correspondencesTrue.emplace_back(worldPoint, imagePoint);
	}

	auto&& [estimateR, estimateT]{estimateTransformRANSAC(correspondencesTrue, K, 0.5)};

	std::println("Real rotation:");
	std::cout << R << "\n";

	std::println("Real translation:");
	std::cout << t_ << "\n";

#ifndef _DEBUG

	std::println("\033[1;32m=== Test case 1: no noise, no outliers ===\033[0m");

	std::println("Estimated rotation:");
	std::cout << estimateR << "\n";

	std::println("Estimated translation:");
	std::cout << estimateT << "\n";

	std::println("Direct rotation error: {}", cv::norm(R-estimateR));
	std::println("Direct translation error: {}", cv::norm(t-estimateT));

	auto ptsNoisy{ptsTrue
		| std::views::transform([&](auto&& pt2d){return cv::Point2d{pt2d.x + noise(gen), pt2d.y + noise(gen)};})
		| std::ranges::to<decltype(ptsTrue)>()
	};

	decltype(correspondencesTrue) correspondencesNoisy{};
	for (auto&& [worldPoint, imagePoint] : std::views::zip(scenePoints, ptsNoisy)) {
		correspondencesNoisy.emplace_back(worldPoint, imagePoint);
	}

	auto&& [rn, rt]{estimateTransformRANSAC(correspondencesNoisy, K, 3.)};
	estimateR = std::move(rn);
	estimateT = std::move(rt);


	std::println("\033[1;32m=== Test case 2: Gaussian noise, no outliers ===\033[0m");

	std::println("Estimated rotation:");
	std::cout << estimateR << "\n";

	std::println("Estimated translation:");
	std::cout << estimateT << "\n";

	std::println("Direct rotation error: {}", cv::norm(R-estimateR));
	std::println("Direct translation error: {}", cv::norm(t-estimateT));

	auto ptsOutliers{ptsNoisy
	| std::views::transform([&](auto&& p2d){
			if (prob(gen) < 0.3) {
				return cv::Point2d{outlier(gen), outlier(gen)};
			}
			return p2d;
			})
	| std::ranges::to<decltype(ptsTrue)>()
	};

	decltype(correspondencesTrue) correspondencesOutliers{};
	for (auto&& [worldPoint, imagePoint] : std::views::zip(scenePoints, ptsOutliers)) {
		correspondencesOutliers.emplace_back(worldPoint, imagePoint);
	}

	auto&& [ro, to]{estimateTransformRANSAC(correspondencesOutliers, K, 3.)};
	estimateR = std::move(ro);
	estimateT = std::move(to);


	std::println("\033[1;32m=== Test case 3: Gaussian noise, 30% outliers ===\033[0m");

	std::println("Estimated rotation:");
	std::cout << estimateR << "\n";

	std::println("Estimated translation:");
	std::cout << estimateT << "\n";	

	std::println("Direct rotation error: {}", cv::norm(R-estimateR));
	std::println("Direct translation error: {}", cv::norm(t-estimateT));
#endif

	std::vector<cv::Point3d> scenePointsDegenerate{{
		{0.0,  0.0,  5.0},
		{1.0,  0.0,  5.0},
		{-1.0, 0.0,  5.0},
		{0.0,  1.0,  5.0},
		{0.0, -1.0,  5.0},
		{1.0,  1.0,  5.0},
		{-1.0, 1.0,  5.0},
		{1.0, -1.0,  5.0},
		{-1.0,-1.0,  5.0},
		{2.0,  0.0,  5.0},
		{-2.0, 0.0,  5.0},
		{0.0,  2.0,  5.0},
		{0.0, -2.0,  5.0},
		{2.0,  2.0,  5.0},
		{-2.0,-2.0,  5.0},
	}};

	auto ptsDegenerate{hw3::projectPoints(scenePointsDegenerate, P)};
	decltype(correspondencesTrue) correspondencesDegenerate{};
	for (auto&& [worldPoint, imagePoint] : std::views::zip(scenePointsDegenerate, ptsDegenerate)) {
		correspondencesDegenerate.emplace_back(worldPoint, imagePoint);
	}


	std::println("\033[1;32m=== Test case 4: degenerate scene, no noise, no outliers ===\033[0m");
	auto&& [rdg, tdg]{estimateTransformRANSAC(correspondencesDegenerate, K, 3.)};
	estimateR = std::move(rdg);
	estimateT = std::move(tdg);


	std::println("Estimated rotation:");
	std::cout << estimateR << "\n";

	std::println("Estimated translation:");
	std::cout << estimateT << "\n";	

	std::println("Direct rotation error: {}", cv::norm(R-estimateR));
	std::println("Direct translation error: {}", cv::norm(t-estimateT));



	return 0;
}

