#include <ranges>
#include <print>
#include <algorithm>
#include <random>
#include <cassert>
#include "hw3utils.hpp"

	
namespace {

	struct square {
		template<typename T>
		constexpr decltype(auto) operator()(T&& x) const {
			return x * x;
		}
	};


	constexpr inline double linearError(double x, double y, double k, double b) {
		return std::abs(y-k*x-b);
	}
}


std::pair<double, double> hw3::minimizeMSE(const std::vector<double>& X, const std::vector<double>& Y) {
	const auto n{X.size()};

	const double sumXY{std::ranges::fold_left(
			std::views::zip(X, Y), 0.,
			[](double curr, const auto&& pair) {
				auto&& [x, y]{pair};
				return curr + x*y;
			})};
	const double sumX{std::ranges::fold_left(X, 0., std::plus{})};
	const double sumY{std::ranges::fold_left(Y, 0., std::plus{})};
	const double sumX2{std::ranges::fold_left(X | std::views::transform(square{}), 0.0,
			std::plus{})};

	const double k{(n * sumXY - sumX * sumY)/(n*sumX2-sumX*sumX)};
	const double b{(sumY-k*sumX)/n};

	// std::println("sumXY={}", sumXY);
	// std::println("sumX={}", sumX);
	// std::println("sumY={}", sumY);
	// std::println("sumX2={}", sumX2);

	return {k, b};
}

std::pair<double, double> hw3::linearEstimateRANSAC(const std::vector<std::pair<double, double>>& points,
		double tau, double p, std::size_t k) {
	static std::random_device rd{};
	static std::mt19937 gen{rd()};

	const auto pointCount{points.size()};

	constexpr double w{0.5};
						
	const auto N{static_cast<std::size_t>(std::log(1-p)/std::log(1-w*w))};

	double bestK{}, bestB{};
	double bestRatio{};

	for (auto i{0uz}; i < N; i++) { // N -> 5
		std::vector<std::pair<double, double>> sample{};

		decltype(i) inlierPointCount{};

		std::sample(std::ranges::begin(points),
					std::ranges::end(points),
					std::back_inserter(sample),
					k, gen);

		std::vector<double> X{sample | std::views::elements<0> | std::ranges::to<decltype(X)>()};
		std::vector<double> Y{sample | std::views::elements<1> | std::ranges::to<decltype(Y)>()};

		// std::println("X = {}", X);
		// std::println("Y = {}", Y);

		auto&& [currK, currB]{minimizeMSE(X, Y)};

		// std::println("{}, {}", currK, currB);

		for (const auto& [x, y]: points) {
			if (linearError(x, y, currK, currB) < tau) {
				inlierPointCount++;
			}
		}
		
		const double currRatio{static_cast<double>(inlierPointCount) / pointCount};
		if (currRatio > bestRatio) {
			bestRatio = currRatio;
			bestK = currK;
			bestB = currB;
		}
	}

	std::vector<double> inlierX{}, inlierY{};

	for (const auto& [x, y] : points) {
		if (linearError(x, y, bestK, bestB) < tau) {
			inlierX.push_back(x);
			inlierY.push_back(y);
		}
	}
	auto&& [refinedK, refinedB]{minimizeMSE(inlierX, inlierY)};
	return {refinedK, refinedB};
}

#ifdef USE_OPENCV
std::vector<cv::Point2d> hw3::projectPoints(const std::vector<cv::Point3d>& points3D, const cv::Matx34d& projectionMat) {

	std::vector<cv::Point2d> res{};

	for (const auto& p3d : points3D) {
		cv::Vec4d X{p3d.x, p3d.y, p3d.z, 1.0};

		cv::Vec3d x1{projectionMat * X};
		res.emplace_back(x1(0)/x1(2), x1(1)/x1(2));
	}

	return res;
}

namespace {


	cv::Point2d projectPoint(const cv::Point3d& pt, const cv::Matx34d& P) {
		cv::Vec4d X{pt.x, pt.y, pt.z, 1.};
		cv::Vec3d h{P*X};
		return { h[0]/h[2], h[1]/h[2] };
	}


	double reprojectionError(const cv::Point3d& truePoint, const cv::Point2d& projectedPoint,
			const cv::Matx33d& R, const cv::Vec3d& t, const cv::Matx33d& K) {
		cv::Matx34d Rt{};
		cv::hconcat(R, t, Rt);
		cv::Matx34d P{K*Rt};
		return cv::norm(projectedPoint - projectPoint(truePoint, P));
	}


	std::pair<cv::Matx33d, cv::Vec3d> directLinearTransform(const std::vector<Correspondence>& correspondences, const cv::Matx33d& K) {

		assert(correspondences.size() >= 6);

		std::array<std::array<double, 12>, 12> A{};

		for (auto i{0uz}; i < 12; i += 2) {
			double X{correspondences[i/2].worldPoint.x};
			double Y{correspondences[i/2].worldPoint.y};
			double Z{correspondences[i/2].worldPoint.z};
			double u{correspondences[i/2].imagePoint.x};
			double v{correspondences[i/2].imagePoint.y};

			A[i][0] = X; 
			A[i][1] = Y; 
			A[i][2] = Z;
			A[i][3] = 1.;
			A[i][4] = 0.;
			A[i][5] = 0.;
			A[i][6] = 0.;
			A[i][7] = 0.;
			A[i][8] = -u * X;
			A[i][9] = -u * Y; 
			A[i][10] = -u * Z;
			A[i][11] = -u;

			A[i+1][0] = 0.; 
			A[i+1][1] = 0.; 
			A[i+1][2] = 0.;
			A[i+1][3] = 0.;
			A[i+1][4] = X;
			A[i+1][5] = Y;
			A[i+1][6] = Z;
			A[i+1][7] = 1.;
			A[i+1][8] = -v * X;
			A[i+1][9] = -v * Y; 
			A[i+1][10] = -v * Z;
			A[i+1][11] = -v;
		}
		cv::Mat AMat(12, 12, CV_64F);
		std::memcpy(AMat.ptr<double>(), A.data()->data(), 12 * 12 * sizeof(double));

		cv::Mat W{}, U{}, Vt{};
		cv::SVD::compute(AMat, W, U, Vt, cv::SVD::FULL_UV);

		cv::Mat T{Vt.row(11)};
		cv::Mat T_{T.reshape(1, 3).clone()};
		cv::Matx34d P{T_.ptr<double>()};
		cv::Matx34d Rt{K.inv()*P};
		cv::Matx33d R{
			Rt(0,0), Rt(0,1), Rt(0,2),
			Rt(1,0), Rt(1,1), Rt(1,2),
			Rt(2,0), Rt(2,1), Rt(2,2)
		};
		cv::Vec3d t{Rt(0,3), Rt(1,3), Rt(2,3)};
		return {R, t};
	}

	
}


struct hw3::Correspondence {
	cv::Point3d worldPoint;
	cv::Point2d imagePoint;
};


std::pair<cv::Matx33d, cv::Vec3d> hw3::estimateTransformRANSAC(const std::vector<Correspondence>& correspondences, const cv::Matx33d& K, double tau, double p=0.99) {

	static std::random_device rd{};
	static std::mt19937 gen{rd()};

	const auto pointCount{points.size()};

	constexpr double w{0.5};
	const auto N{static_cast<std::size_t>(std::log(1-p)/std::log(1-w*w))};

	cv::Matx33d bestR{};
	cv::Vec3d bestT{};

	for (auto i{0uz}; i < N; i++) {
		std::vector<Correspondence> sample{};
		
		decltype(i) inlierPointCount{};

		std::sample(correspondences.begin(),
					correspondences.end(),
					6, gen);
		
	}
}
#endif





