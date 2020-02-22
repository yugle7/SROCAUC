#pragma once

#include <catboost/private/libs/data_types/pair.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/containers/2d_array/2d_array.h>
#include <library/fast_exp/fast_exp.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/string/split.h>
#include <util/system/yassert.h>

#include <cmath>

#include "types.h"

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

    static bool Greater(const TExample& a, const TExample& b) {
        return a.Approx != b.Approx ? a.Approx < b.Approx : a.Size > b.Size;
    }

    static bool IdealGreater(const TExample& a, const TExample& b) {
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

class TSizedAucError final : public IDerCalcer {
    static constexpr ui32 ItersCount = 100;

public:
    TSizedAucError(const TMap<TString, TString>& params, bool isExpApprox)
        : IDerCalcer(isExpApprox, 1, EErrorType::QuerywiseError) {

        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    float CalcQueryError(TVector<TExample>& sample) const {
        Sort(sample.begin(), sample.end(), TExample::Greater);
        auto auc = GetAuc(sample);

        Sort(sample.begin(), sample.end(), TExample::IdealGreater);
        auto idealAuc = GetAuc(sample);

        return auc / idealAuc;
    }

    void CalcDersForQueries(
        int queryStartIndex,
        int queryEndIndex,
        const TVector<double>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        TArrayRef<TDers> ders,
        ui64,
        NPar::TLocalExecutor* localExecutor) const {

        ui32 offset = queriesInfo[queryStartIndex].Begin;

        NPar::ParallelFor(
            *localExecutor,
            queryStartIndex,
            queryEndIndex,
            [&](ui32 queryIndex) {
                CalcQueryDers(offset, approx, target, weight, ders, queriesInfo[queryIndex]);
            });
    }

    void CalcQueryDers(
        int offset,
        const TVector<double>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        TArrayRef<TDers> ders,
        TQueryInfo query) const {

        auto querySize = query.End - query.Begin;
        auto dersBegin = ders.begin() + query.Begin - offset;

        Fill(dersBegin, dersBegin + querySize, TDers{0, 0, 0});

        auto sample = GetSample(approx, target, weight, offset, querySize);

        Sort(sample.begin(), sample.end(), TExample::Greater);
        Sum(sample);

        for (ui32 iter = 0; iter < ItersCount; ++iter) {
            Shuffle(sample.begin(), sample.end());

            for (ui32 i = 1; i < querySize; ++i) {
                auto a = sample[i - 1];
                auto b = sample[i];

                if (a.SumTargets > b.SumTargets) {
                    std::swap(a, b);
                }
                auto deltaAuc = (b.Target - a.Target) * b.Size - (b.SumTargets - a.SumTargets) * (b.Size - a.Size);
                auto deltaApprox = a.Approx - b.Approx;

                auto sigma = Sigmoid(deltaAuc > 0 ? deltaApprox : -deltaApprox);
                auto deltaDer = query.Weight * sigma * sigma * deltaAuc;

                ders[a.Id].Der1 -= deltaDer;
                ders[b.Id].Der1 += deltaDer;
            }
        }
    }
};
