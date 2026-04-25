#include <print>
#include <random>
#include <algorithm>
#include <ranges>
#include "hw3utils.hpp"


int main(int argc, char* argv[]) {
	constexpr double p{.99};
	constexpr double tau{5.};
	
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

	// std::println("Ground truth:");
	//    for (auto i{0uz}; i < 50; i++)
	//    {
	//        std::println("{}: {}", x[i], y[i]);
	//    }
	std::println("True parameters:");
	std::println("k = {}, b = {}", k, b);

	// const std::vector<double> xTest{0., 1.};
	// const std::vector<double> yTest{0., 1.};
	// const auto&& [testK, testB]{hw3::minimizeMSE(xTest, yTest)};
	// std::println("Estimated test (true=1,0): {}, {}", testK, testB);

	std::vector<std::pair<double, double>> points{std::views::zip(x, y) | std::ranges::to<decltype(points)>()};

	const auto&& [guessedK, guessedB]{hw3::linearEstimateRANSAC(points, tau, p)};

	std::println("Guessed parameters: ");
	std::println("k^ = {}, b^ = {}", guessedK, guessedB);

	return 0;
}
