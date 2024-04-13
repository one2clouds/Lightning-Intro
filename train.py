import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.profilers import PyTorchProfiler



if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    strategy = DeepSpeedStrategy()

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    

    )