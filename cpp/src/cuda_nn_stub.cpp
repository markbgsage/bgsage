#include "bgbot/cuda_nn.h"

namespace bgbot {

bool cuda_available() {
    return false;
}

SupervisedTrainResult cuda_supervised_train(const SupervisedTrainConfig& config) {
    SupervisedTrainResult result;
    result.best_score = 1e9;
    result.best_epoch = -1;
    result.epochs_completed = 0;
    result.total_seconds = 0.0;
    return result;
}

} // namespace bgbot
