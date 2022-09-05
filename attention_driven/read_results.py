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
        for learning_rate, result in sorted(results.items()):
            test_score = result.metrics["test_score"]
            print("  ", learning_rate, f"{test_score:.2f}")


if __name__ == "__main__":
    main()
