import argparse
import numpy as np
import onnxruntime as ort
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Run inference with an ONNX model")
    p.add_argument("--model", default="/home/elias/PROJECT/AICryptoPredictor/results/pipeline2_model.onnx",
                   help="Path to ONNX model file")
    p.add_argument("--npy", type=str, default=None,
                   help="Path to .npy file containing input features (shape [N,14] or [14])")
    p.add_argument("--features", type=float, nargs='*', default=None,
                   help="Inline 14 float features, e.g. --features 0.1 0.2 ... 14 values")
    p.add_argument("--batch", type=int, default=1, help="Batch size if using random input")
    p.add_argument("--input_name", type=str, default="input", help="Input tensor name in model")
    p.add_argument("--sigmoid_threshold", type=float, default=0.789, help="Classification threshold")
    return p.parse_args()

def load_input(args):
    if args.npy:
        x = np.load(args.npy)
        if x.ndim == 1:
            x = x.reshape(1, -1)
    elif args.features:
        vals = np.array(args.features, dtype=np.float32)
        if vals.size != 14:
            print("Expected 14 features with --features", file=sys.stderr)
            sys.exit(1)
        x = vals.reshape(1, 14)
    else:
        # Random input for quick test
        x = np.random.rand(args.batch, 14).astype(np.float32)
    x = x.astype(np.float32)
    return x

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    x = load_input(args)

    outputs = sess.run(None, {args.input_name: x})
    prob = outputs[0]  # expected shape [N,1]
    print("Probabilities:", prob.squeeze())

    # Optionnel: classification binaire avec seuil
    preds = (prob > args.sigmoid_threshold).astype(int)
    print("Predictions:", preds.squeeze())

if __name__ == "__main__":
    main()