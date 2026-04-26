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


	std::pair<cv::Matx33d, cv::Vec3d> directLinearTransform(const std::vector<hw3::Correspondence>& correspondences, const cv::Matx33d& K) {
		assert(correspondences.size() >= 6);
		const auto n{correspondences.size()};

		cv::Mat A(2*n, 12, CV_64F, 0.);

		for (auto i{0uz}; i < n; i++) {
			double X{correspondences[i].worldPoint.x};
			double Y{correspondences[i].worldPoint.y};
			double Z{correspondences[i].worldPoint.z};
			double u{correspondences[i].imagePoint.x};
			double v{correspondences[i].imagePoint.y};
			
			double* row0 = A.ptr<double>(2*i);
			double* row1 = A.ptr<double>(2*i+1);
			
			row0[0]=X;  row0[1]=Y;  row0[2]=Z;  row0[3]=1.;
			row0[4]=0.; row0[5]=0.; row0[6]=0.; row0[7]=0.;
			row0[8]=-u*X; row0[9]=-u*Y; row0[10]=-u*Z; row0[11]=-u;

			row1[0]=0.; row1[1]=0.; row1[2]=0.; row1[3]=0.;
			row1[4]=X;  row1[5]=Y;  row1[6]=Z;  row1[7]=1.;
			row1[8]=-v*X; row1[9]=-v*Y; row1[10]=-v*Z; row1[11]=-v;
		}

		cv::Mat W{}, U{}, Vt{};
		cv::SVD::compute(A, W, U, Vt, cv::SVD::FULL_UV);

		cv::Mat T{Vt.row(11)};
		cv::Mat T_{T.reshape(1, 3).clone()};
		cv::Matx34d P{T_.ptr<double>()};
		cv::Matx34d Rt{K.inv()*P};

		const double s{cv::norm(cv::Vec3d(Rt(0,0), Rt(1,0), Rt(2,0)))};

		cv::Matx33d R{
			Rt(0,0), Rt(0,1), Rt(0,2),
			Rt(1,0), Rt(1,1), Rt(1,2),
			Rt(2,0), Rt(2,1), Rt(2,2)
		};
		cv::Vec3d t{Rt(0,3)/s, Rt(1,3)/s, Rt(2,3)/s};

		cv::Mat RProj_{R}, Uw{}, Sw{}, Vwt{};
		cv::SVD::compute(RProj_, Sw, Uw, Vwt);
		cv::Matx33d RProj{cv::Matx33d{cv::Mat(Uw * Vwt)}};
		
		if (cv::determinant(RProj) < 0) {
			RProj *= -1.;
			t *= -1.;
		}

#ifdef _DEBUG
		std::println("Reconstructed rotation:");
		std::cout << RProj << "\n";
		std::println("Reconstructed translation:");
		std::cout << t << "\n";
#endif
		
		return {RProj, t};
	}


	inline std::size_t calculateNRANSAC(double w, double p=.99, double n=6.) {
		w = std::clamp(w, 1e-9, 1.-1e-9);
		return static_cast<std::size_t>(std::log(1-p)/std::log(1-std::pow(w, n)));
	}
	
}


std::pair<cv::Matx33d, cv::Vec3d> hw3::estimateTransformRANSAC(const std::vector<hw3::Correspondence>& correspondences, const cv::Matx33d& K, double tau, double p) {

	static std::random_device rd{};
	static std::mt19937 gen{rd()};

	const auto pointCount{correspondences.size()};

	double w{0.5};
	auto N{calculateNRANSAC(w)};

	cv::Matx33d bestR{};
	cv::Vec3d bestT{};
	double bestRatio{};
	
	auto i{0uz};
	while (i < N) {
		std::vector<Correspondence> sample{};
		
		decltype(i) inlierPointCount{};

		std::sample(correspondences.begin(),
					correspondences.end(),
					std::back_inserter(sample),
					6, gen);
#ifdef _DEBUG
		std::println("Sample size is {}", sample.size() );
#endif
		const auto&& [currR, currT]{directLinearTransform(sample, K)};

		for (auto&& [worldPoint, imagePoint]: correspondences) {
			const double err = reprojectionError(worldPoint, imagePoint, currR, currT, K);
			// std::println("Rotation estimate:");
			// std::cout << currR << "\n";
			// std::println("Translation estimate:");
			// std::cout << currT << "\n";
			// std::println("Reprojection error: {}", err);
			if (err < tau) {
				inlierPointCount++;
			}
		}

		const double currRatio{static_cast<double>(inlierPointCount)/pointCount};
		// std::println("currRatio = {}", currRatio);
		if (currRatio > bestRatio) {
			bestR = currR;
			bestT = currT;
			bestRatio = currRatio;
			N = calculateNRANSAC(bestRatio);
		}	
		i++;
	}
	
	std::vector<Correspondence> inliers{};
	for (const auto& c: correspondences) {
		if (reprojectionError(c.worldPoint, c.imagePoint, bestR, bestT, K) < tau) {
			inliers.push_back(c);
		}
	}
	
	auto&& [refinedR, refinedT]{directLinearTransform(inliers, K)};
	return {refinedR, refinedT};
}


void hw3::testReprojection(const cv::Point3d& pt3d, const cv::Point2d& pt2d, const cv::Matx33d& R, const cv::Vec3d& t, const cv::Matx33d& K) {
	const auto err{reprojectionError(pt3d, pt2d, R, t, K)};
	std::println("Error = {}", err);
}

#endif





