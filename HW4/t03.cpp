#include <ceres/ceres.h>
#include <ceres/rotation.h> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <print>

struct ReprojectionError {
	using RotationType = std::array<double, 9>;
	using TranslationType = std::array<double, 3>;

	ReprojectionError(double observedU, double observedV, bool isC2,
			const RotationType& R_C1_C2, const TranslationType& t_C1_C2,
			const RotationType& K1, const RotationType& K2)
		: observedU(observedU), observedV(observedV), isC2(isC2),
		R_C1_C2(R_C1_C2), t_C1_C2(t_C1_C2), K1(K1), K2(K2)
		{}

	template<typename T>
	bool operator()(const T* const r, const T* const t, const T* const guessedPoint, T* residuals) const {
		T p[3]{};
		ceres::AngleAxisRotatePoint(r, guessedPoint, p);
		p[0] += t[0];
		p[1] += t[1];
		p[2] += t[2];

		if (isC2) {
			T pC2[3]{};
			pC2[0] = T{R_C1_C2[0]}*p[0] + T{R_C1_C2[1]}*p[1] + T{R_C1_C2[2]}*p[2] + T{t_C1_C2[0]};
			pC2[1] = T{R_C1_C2[3]}*p[0] + T{R_C1_C2[4]}*p[1] + T{R_C1_C2[5]}*p[2] + T{t_C1_C2[1]};
			pC2[2] = T{R_C1_C2[6]}*p[0] + T{R_C1_C2[7]}*p[1] + T{R_C1_C2[8]}*p[2] + T{t_C1_C2[2]};
			p[0] = pC2[0];
			p[1] = pC2[1];
			p[2] = pC2[2];
		}
		
		const T x{p[0]/p[2]};
		const T y{p[1]/p[2]};

		T u{};
		T v{};
		
		double fx{}, cx{}, fy{}, cy{};
		 
		fx = (!isC2 ? K1 : K2)[0];
		fy = (!isC2 ? K1 : K2)[4];
		cx = (!isC2 ? K1 : K2)[2];
		cy = (!isC2 ? K1 : K2)[5];

		u = T{fx} * x + T{cx};
		v = T{fy} * y + T{cy};
		residuals[0] = u - T{observedU};
		residuals[1] = v - T{observedV};

		return true;
	}

	static ceres::CostFunction* Create(const double observedU, const double observedV, bool isC2,
			const RotationType& R_C1_C2, const TranslationType& t_C1_C2,
			const RotationType& K1, const RotationType& K2) {
		return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(
				new ReprojectionError(observedU, observedV, isC2, R_C1_C2, t_C1_C2, K1, K2)
				);
	}
	
	double observedU, observedV;
	bool isC2;
	std::array<double, 9> R_C1_C2;
	std::array<double, 3> t_C1_C2;
	std::array<double, 9> K1, K2;
};


std::vector<cv::Point2d> projectPoints(const std::vector<cv::Point3d>& points3D, const cv::Matx34d& projectionMat) {
	std::vector<cv::Point2d> res{};

	for (const auto& p3d : points3D) {
		cv::Vec4d X{p3d.x, p3d.y, p3d.z, 1.0};
	
		cv::Vec3d x1{projectionMat * X};
		res.emplace_back(x1(0)/x1(2), x1(1)/x1(2));
	}

	return res;
}


int main(int argc, char* argv[]) {
	/* Points (frame=W) */
	const std::vector<cv::Point3d> points3D{{0.0,  0.0,  5.0},
		{1.0, 0.5, 6.0},
		{-1.0, 1.0, 7.0},
		{2.0, -1.0, 5.5},
		{-2.0, -0.5, 6.5},
		{0.5, 2.0, 8.0},
		{-1.5, -2.0, 7.5},
		{1.5, 1.5, 9.0}
	};

	/* Camera matrices (frame=W) */
    constexpr double fx{300.}, fy{300.}, cx{320.}, cy{320.};
    const cv::Matx33d K1{fx, 0., cx, 0., fy, cy, 0., 0., 1.};
    const cv::Matx33d K2{fx, 0., cx, 0., fy, cy, 0., 0., 1.};

	/* Poses relative to W, that is: W->C1, W->C2 */
	const cv::Vec3d r1Vec{0.25, 0.33, 0.8};
	cv::Mat R_W_C1_{};
	cv::Rodrigues(r1Vec, R_W_C1_);
	const cv::Matx33d R_W_C1{R_W_C1_};
	const cv::Vec3d camera1Coords{3.0, 3.0, 3.0};
	const cv::Vec3d t_W_C1{-R_W_C1 * camera1Coords};
	cv::Matx34d T_W_C1{};
	cv::hconcat(R_W_C1, t_W_C1, T_W_C1);
	const cv::Matx34d P1A{K1*T_W_C1}; /* P1A is now the projection matrix for points in W onto the plane in C1 */


	const cv::Vec3d r2Vec{0.20, 0.25, 0.6};
	cv::Mat R_W_C2_{};
	cv::Rodrigues(r2Vec, R_W_C2_);
	const cv::Matx33d R_W_C2{R_W_C2_};
	const cv::Vec3d camera2Coords{4.0, 4.0, 4.0};
	const cv::Vec3d t_W_C2{-R_W_C2 * camera2Coords};
	cv::Matx34d T_W_C2{};
	cv::hconcat(R_W_C2, t_W_C2, T_W_C2); 
	const cv::Matx34d P2A{K2*T_W_C2}; /* P2A is now the projection matrix for points in W onto the plane in C2 */

	/* Computing the constant rigid transform C1->C2 */
	const cv::Matx33d R_C1_C2{R_W_C2 * R_W_C1.t()};
	const cv::Vec3d t_C1_C2{t_W_C2 - R_C1_C2 * t_W_C1};
		
	/* True transformation between points A and B
	   A is meant to be the origin of W and B some other point */
	const cv::Vec3d rWVec{0.1, 0.1, 0.1};
	cv::Mat R_W_{};
	cv::Rodrigues(rWVec, R_W_);
	const cv::Matx33d R_W{R_W_};
	const cv::Vec3d BCoords{0.5, 0.5, 0.5};
	const cv::Vec3d t_W{-R_W * BCoords};
	cv::Matx34d T_W{};
	cv::hconcat(R_W, t_W, T_W);

	/* Recomputing extrinsics in the new point B */
	const cv::Vec3d camera1CoordsB{R_W*camera1Coords + t_W};
	const cv::Matx33d R_W_C1_B{R_W_C1 * R_W.t()};
	const cv::Vec3d t_W_C1_B{-R_W_C1_B*camera1CoordsB};
	cv::Mat rVecC1B_{};
	cv::Rodrigues(R_W_C1_B, rVecC1B_);
	cv::Vec3d rVecC1B{rVecC1B_};
	cv::Matx34d T_W_C1_B{};
	cv::hconcat(R_W_C1_B, t_W_C1_B, T_W_C1_B);
	cv::Matx34d P1B{K1*T_W_C1_B};

	const cv::Vec3d camera2CoordsB{R_W*camera2Coords + t_W};
	const cv::Matx33d R_W_C2_B{R_W_C2 * R_W.t()};
	cv::Mat rVecC2B_{};
	cv::Rodrigues(R_W_C2_B, rVecC2B_);
	cv::Vec3d rVecC2B{rVecC2B_};
	const cv::Vec3d t_W_C2_B{-R_W_C2_B*camera2CoordsB};
	cv::Matx34d T_W_C2_B{};
	cv::hconcat(R_W_C2_B, t_W_C2_B, T_W_C2_B);
	cv::Matx34d P2B{K2*T_W_C2_B};


	/* Finding the projections */
	const auto pointsCam1A{projectPoints(points3D, P1A)};
	const auto pointsCam2A{projectPoints(points3D, P2A)};
	const auto pointsCam1B{projectPoints(points3D, P1B)};
	const auto pointsCam2B{projectPoints(points3D, P2B)};

	/* Setting up the problem */
	ceres::Problem problem{};

	/* Flattening arrays 'cause reasons */

	double RArr[9], K1Arr[9], K2Arr[9], tArr[3];
	std::memcpy(RArr, R_C1_C2.val, sizeof RArr);
	std::memcpy(K1Arr, K1.val, sizeof K1Arr);
	std::memcpy(K2Arr, K2.val, sizeof K2Arr);
	std::memcpy(tArr, t_C1_C2.val, sizeof tArr);

	auto RArr_{std::to_array(RArr)};
	auto tArr_{std::to_array(tArr)};
	auto K1_{std::to_array(K1Arr)};
	auto K2_{std::to_array(K2Arr)};

	double rA[3]{r1Vec[0], r1Vec[1], r1Vec[2]};
	double rB[3]{rVecC1B[0] + 0.1, rVecC1B[1] + 0.1, rVecC1B[2] + 0.1};
	double tA[3]{t_W_C1[0], t_W_C1[1], t_W_C1[2]};
	double tB[3]{t_W_C1_B[0] + 0.5, t_W_C1_B[1] + 0.5, t_W_C1_B[2] + 0.5};

	std::vector<std::array<double, 3>> points3DGuessed{};
	for (const auto& p : points3D) {
	    points3DGuessed.push_back({p.x + 1., p.y - 2., p.z + 3.});
	}

	for (auto i{0uz}; i < points3D.size(); i++) {
		auto* costFunctionC1A{ReprojectionError::Create(pointsCam1A[i].x, pointsCam1A[i].y, false, RArr_, tArr_, K1_, K2_)};
		auto* costFunctionC2A{ReprojectionError::Create(pointsCam2A[i].x, pointsCam2A[i].y, true, RArr_, tArr_, K1_, K2_)};
		auto* costFunctionC1B{ReprojectionError::Create(pointsCam1B[i].x, pointsCam1B[i].y, false, RArr_, tArr_, K1_, K2_)};
		auto* costFunctionC2B{ReprojectionError::Create(pointsCam2B[i].x, pointsCam2B[i].y, true, RArr_, tArr_, K1_, K2_)};

		problem.AddResidualBlock(costFunctionC1A, nullptr, rA, tA, points3DGuessed[i].data());
		problem.AddResidualBlock(costFunctionC2A, nullptr, rA, tA, points3DGuessed[i].data());
		problem.AddResidualBlock(costFunctionC1B, nullptr, rB, tB, points3DGuessed[i].data());
		problem.AddResidualBlock(costFunctionC2B, nullptr, rB, tB, points3DGuessed[i].data());	
	}

	problem.SetParameterBlockConstant(rA);
	problem.SetParameterBlockConstant(tA);

	ceres::Solver::Options options{};
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary{};
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	std::println("Ground truth rotation:");
	std::cout << rVecC1B << "\n";

	std::println("Guessed:");
	for (auto i{0uz}; i < 3; i++){
		std::print("{} ", rB[i]);
	}

	std::println("");

	std::println("Ground truth translation:");
	std::cout << t_W_C1_B << "\n";

	std::println("Guessed:");
	for (auto i{0uz}; i < 3; i++){
		std::print("{} ", tB[i]);
	}

	std::println("");

	for (auto i{0uz}; i < points3DGuessed.size(); i++) {
		std::println("Point {} true: ", i);
		std::cout << points3D[i] << std::endl;
		std::println("Point {} guessed: ", i);
		std::println("{}", points3DGuessed[i]);
	}

	return 0;	
}
