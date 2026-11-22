"""Serve seven zero actions per prediction for LIBERO pipeline tests."""

import numpy as np
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve


class NullPolicy(PredictModelServer):
    def predict(self, obs, row):
        print(f"Received instruction {obs['task_description']!r}", flush=True)
        return {"actions": np.zeros(7, dtype=np.float32)}


if __name__ == "__main__":
    serve(NullPolicy(), host="127.0.0.1", port=8000)
