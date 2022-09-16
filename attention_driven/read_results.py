import os
import pickle

from attention_driven import RESULTS_DIR


def main():
    for experiment_name in os.listdir(RESULTS_DIR):
        if not experiment_name.endswith("Experiment"):
            continue

        results_path = os.path.join(RESULTS_DIR, experiment_name, "predictions")
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        print(experiment_name)
        for learning_rate, result_by_lr in sorted(results.items()):
            print(f"  {learning_rate:e}")
            for split_name, result in result_by_lr.items():
                loss = result.metrics["test_loss"]
                score = result.metrics["test_score"]
                print(f"    {split_name.ljust(10)} | loss {loss:.2f} score {score:.1f}")


if __name__ == "__main__":
    main()
