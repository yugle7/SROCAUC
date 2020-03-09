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

#include "types.h"
#include "sized_roc_auc.h"


namespace NSizedRocAucMetric {

class TSizedAucError final : public IDerCalcer {
    static constexpr ui32 ItersCount = 100;

public:
    TSizedAucError(const TMap<TString, TString>& params, bool isExpApprox)
        : IDerCalcer(isExpApprox, 1, EErrorType::QuerywiseError) {

        CB_ENSURE(isExpApprox == false, "Approx format does not match");
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

        Sort(sample.begin(), sample.end(), TExample::LessByApprox);
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

}
