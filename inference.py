import time
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from model.nets import CQTNet
from model.dataset import InferenceDataset
from model.utils import load_model
from utilities.utils import format_time

# My linux is complaining without the following line
# It required for the DataLoader to have the num_workers > 0
torch.multiprocessing.set_sharing_strategy("file_system")


def collate_single(x):
    """
    Collate function that extracts the single item from a batch.

    Enables multiprocessing to work on MacOS
    """
    return x[0]


@torch.inference_mode()
def main(
    model: CQTNet,
    loader: DataLoader,
    granularity: str,
    overlap: float | None,
    amp: bool,
    fp16_outputs: bool,
):

    if granularity == "chunk":
        assert (
            overlap is not None
        ), "Overlap must be specified for chunk granularity."
        assert 0 <= overlap < 1, "Overlap must be in [0,1)"
        W = int(
            loader.dataset.context_length // loader.dataset.downsample_factor
        )
        H = int(W * (1 - overlap))

    device = next(model.parameters()).device

    model.eval()

    print("Extracting embeddings...")
    if fp16_outputs:
        print("Storing embeddings as float16.")
    t0 = time.monotonic()
    for idx, (cqt, path_audio, path_output) in enumerate(loader):
        if cqt is None:
            continue
        assert cqt.ndim == 2, f"Expected cqt.ndim == 2, got {cqt.ndim}"

        # Reshape the CQT tensor based on granularity
        if granularity == "track":
            cqt = cqt.unsqueeze(0).unsqueeze(1)  # (F,T) -> (1,1,F,T)
        elif granularity == "chunk":
            cqt = cqt.unfold(
                dimension=-1, size=W, step=H
            )  # (F,T) -> (F, N, W)
            cqt = cqt.permute(1, 0, 2)  # (N,F,W)
            cqt = cqt.unsqueeze(1)  # (N,1,F,W)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        # Inference
        try:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                embedding = model(cqt.to(device))  # (N, D)
        except Exception as e:
            print(
                f"Error at {str(path_audio)} during embedding extraction: {repr(e)}"
            )
            continue

        # Convert to numpy and save
        try:
            embedding = embedding.cpu().numpy()
            if fp16_outputs:
                embedding = embedding.astype(np.float16)
            path_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(path_output, embedding)
        except Exception as e:
            print(f"Error at {str(path_audio)} during writing: {repr(e)}")
            continue

        if (idx + 1) % (len(loader) // 20) == 0 or idx == len(loader) - 1:
            print(f"[{(idx+1):>{len(str(len(loader))):,}}/{len(loader):,}]")

    print(f"Total time: {format_time(time.monotonic() - t0)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="""Path to the directory containing the audio files.""",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="""Path to the configuration file of the trained model. 
        The config will be used to find model weigths.""",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="""Path to the output directory. The tree structure of the 
        input directory will be replicated here.""",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["track", "chunk"],
        default="track",
        help="""Embedding granularity level to use for extraction.
        Track-level embeddings create a single embedding for the whole track.
        Chunk-level embeddings create embeddings using the model's context 
        length.""",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="""Overlap ratio to use during chunk-level embedding extraction.""",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="""Store the embeddings with FP16 precision to save space.""",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers to use in the DataLoader.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="""Flag to disable the GPU. If not provided, 
        the GPU will be used if available.""",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="""Flag to disable Automatic Mixed Precision for 
        inference. If not provided, AMP usage will depend on 
        the model config file.""",
    )
    parser.add_argument(
        "--num-partitions",
        default=1,
        help="""Number of partitions to split the input files into. 
        If greater than 1, the script will process only the files
        corresponding to the specified partition.""",
        type=int,
    )
    parser.add_argument(
        "--partition",
        help="""Partition number to process""",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    if args.granularity == "chunk" and args.overlap is None:
        parser.error("--overlap is required for --granularity chunk")

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    print("\033[36m\nExperiment Configuration:\033[0m")
    print(
        "\033[36m"
        + yaml.dump(config, indent=4, width=120, sort_keys=False)
        + "\033[0m"
    )

    if args.no_gpu:
        device = torch.device("cpu")
        # If no GPU is available, disable AMP
        amp = False
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        if args.disable_amp:
            amp = False
        else:
            amp = config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"]
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS supports float16 autocast as of PyTorch 2.1+
        if args.disable_amp:
            amp = False
        else:
            amp = config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"]
    else:
        device = torch.device("cpu")
        amp = False
    print(f"\033[31mDevice: {device}\033[0m")
    print(
        f"\033[31mAutomatic Mixed Precision {'enabled' if amp else 'disabled'}.\033[0m"
    )

    # Load the pretrained model weights
    model = load_model(config, device=device, mode="infer")

    # Build the input output pairs while preserving relative paths
    all_paths = [
        p for ext in ("*.wav", "*.mp3") for p in args.input_dir.rglob(ext)
    ]
    print(f"Found {len(all_paths):,} audio files.")
    path_pairs = []
    for input_path in all_paths:
        output_path = (
            args.output_dir / input_path.relative_to(args.input_dir)
        ).with_suffix(".npy")
        path_pairs.append((input_path, output_path))

    # partition the files
    if args.num_partitions > 1:
        path_pairs = path_pairs[args.partition :: args.num_partitions]
        print(
            f"Partition {args.partition} will process {len(path_pairs):,} files"
        )

    # Create the dataset and the DataLoader
    dataset = InferenceDataset(
        path_pairs,
        context_length=config["TRAIN"]["CONTEXT_LENGTH"],
        downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        #        collate_fn=lambda x: x[0],
        collate_fn=collate_single,
    )

    main(
        model,
        loader,
        granularity=args.granularity,
        overlap=args.overlap,
        amp=amp,
        fp16_outputs=args.fp16,
    )

    #############
    print("Done!")
