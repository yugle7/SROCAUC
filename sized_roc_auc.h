#pragma once

#include <util/generic/ymath.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>

#include <cmath>
#include "types.h"

namespace NSizedRocAucMetric {

struct TExample {
    ui32 Id;
    double Approx;

    float Target;
    float Size;

    float SumTargets = 0;
    float SumSizes = 0;

    TExample(ui32 id, double approx, float target, float size)
        : Id(id), Approx(approx), Target(target), Size(size) {
    }

    static bool LessByApprox(const TExample& a, const TExample& b) {
        return a.Approx != b.Approx ? a.Approx < b.Approx : a.Size > b.Size;
    }

    static bool LessByTarget(const TExample& a, const TExample& b) {
        return a.Target * b.Size < b.Target * a.Size;
    }
};


inline void Sum(TVector<TExample>& sample) {
    float sumTargets = 0;
    float sumSizes = 0;

    for (auto& example : sample) {
        sumTargets += example.Target;
        example.SumTargets = sumTargets;
        sumSizes += example.Size;
        example.SumSizes = sumSizes;
    }
    for (auto& example : sample) {
        example.SumTargets /= sumTargets;
        example.SumSizes /= sumSizes;
    }
}

inline TVector<TExample> GetSample(
    const TVector<double>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    ui32 offset, ui32 querySize) {

    TVector<TExample> sample(querySize);

    for (ui32 id = offset; id < offset + querySize; ++id) {
        sample.emplace_back(id, target[id], approx[id], weight[id]);
    }
    return sample;
}

inline float GetAuc(const TVector<TExample>& sample) {
    float auc = 0;

    float sumSizes = 0;
    float sumTargets = 0;

    for (const auto& example : sample) {
        sumSizes += example.Size;
        sumTargets += example.Target;
        auc += example.Size * sumTargets;
    }
    return auc / sumSizes / sumTargets;
}

inline double Sigmoid(double delta) {
    return delta > 16 ? 1 : (delta < -16 ? 0 : 1 / (1 + std::exp(-delta)));
}

inline float CalcQueryError(TVector<TExample>& sample) {
    Sort(sample.begin(), sample.end(), TExample::LessByApprox);
    auto approxAuc = GetAuc(sample);

    Sort(sample.begin(), sample.end(), TExample::LessByTarget);
    auto targetAuc = GetAuc(sample);

    return approxAuc / targetAuc;
}


inline float CalcQueryError(
    const TVector<double>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    ui32 offset, ui32 querySize) {
    auto sample = GetSample(approx, target, weight, offset, querySize);
    return CalcQueryError(sample);
}

}
