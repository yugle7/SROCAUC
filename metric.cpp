#pragma once

#include "types.h"

#include "sized_roc_auc.h"

/* SizedRocAucMetric */

namespace {

class TSizedRocAucMetric : public TAdditiveMetric<TSizedRocAucMetric> {
public:
    explicit TSizedRocAucMetric(const TMap<TString, TString>& params);

    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int queryBegin,
        int queryEnd
    ) const;

    EErrorType GetErrorType() const override;

    TString GetDescription() const override;

    void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

private:
    double Alpha;
};

}

THolder <IMetric> MakeSizedRocAucMetric(const TMap<TString, TString>& params) {
    return MakeHolder<TSizedRocAucMetric>(params);
}

TSizedRocAucMetric::TSizedRocAucMetric(const TMap<TString, TString>& params) : Alpha(0) {
    if (params.contains("alpha")) {
        Alpha = FromString<float>(params.at("alpha"));
    }
    UseWeights.MakeIgnored();
}

TMetricHolder TSizedRocAucMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryBegin,
    int queryEnd
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(2);
    for (int queryIndex = queryBegin; queryIndex < queryEnd; ++queryIndex) {
        auto begin = queriesInfo[queryIndex].Begin;
        auto end = queriesInfo[queryIndex].End;
        auto querySize = end - begin;

        error.Stats[0] = NSizedRocAucMetric::CalcQueryError( approx[0], target, weight, querySize, begin);
        error.Stats[1] = 1;
    }
    CB_ENSURE(error.Stats[0] > 0, "error calc");
    return error;
}

EErrorType TSizedRocAucMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TSizedRocAucMetric::GetDescription() const {
    return "TSizedRocAucMetric";
}

void TSizedRocAucMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}
