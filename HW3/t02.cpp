#include <print>
#include <algorithm>
#include <random>
#include <ceres/ceres.h>
#include "hw3utils.hpp"

struct LinearError {

	template<typename T>
	bool operator()(const T* parameters, T* cost) const {
		cost[0] = T{y} - parameters[0] * T{x} - parameters[1]; // y - kx-b
		return true;
	}


	static ceres::CostFunction* create(double x, double y) {
		constexpr int kNumParameters{2};
		auto* functor{new LinearError{}};
		functor->x = x;
		functor->y = y;
		return new ceres::AutoDiffCostFunction<LinearError, 1, kNumParameters>{functor};
	}

	double x, y;
};


int main(int argc, char* argv[]) {
	constexpr double k{5.8};
	constexpr double b{0.44};

    std::random_device dev{};
    std::mt19937 rng{dev()}; 
    std::uniform_real_distribution<double> dist{-10., 10.};
    std::normal_distribution<double> noise{0., 1.};
	std::uniform_real_distribution<double> prob{0.0, 1.0};
	std::uniform_real_distribution<double> outlierY{-100.0, 100.0};

	const auto genX = [&](){
		return dist(rng);
	};

	std::vector<double> x(50);
	std::generate(x.begin(), x.end(), genX);

	// HACK
	std::size_t i{0};
	const auto genY = [&](){
		return prob(rng) < 0.3 ? i++, outlierY(rng) : k * x[i++] + b + noise(rng);
	};

	std::vector<double> y(50);
	std::generate(y.begin(), y.end(), genY);

	double guess[2]{1., 0.};

	ceres::Problem problem{};

	for (auto i{0uz}; i < x.size(); i++) {
		problem.AddResidualBlock(
				LinearError::create(x[i], y[i]),
				new ceres::CauchyLoss(1.),
				guess
				);
	}

	ceres::Solver::Options options{};
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary{};
	ceres::Solve(options, &problem, &summary);


	std::cout << summary.FullReport() << "\n";
	std::println("True parameters:");
	std::println("k = {}, b = {}", k, b);
	std::println("Estimated parameters:");
	std::println("k = {}, b = {}", guess[0], guess[1]);

	return 0;
}
