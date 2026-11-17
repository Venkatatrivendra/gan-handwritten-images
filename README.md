# GAN-Based Handwritten Digit Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST dataset and exposes it through a modern Streamlit dashboard.

## Features
- DCGAN generator and discriminator implemented in PyTorch
- MNIST dataset integration with automatic download
- Real training loop with saved checkpoints and training history
- Interactive dashboard with navigation to dashboard, training, generation, analytics, and dataset explorer pages
- Plotly visualizations for loss curves and training details
- Generated image gallery with download support
- Beginner-friendly project guide, architecture overview, and dataset explorer
- Separate MNIST CNN for uploading, preprocessing, and recognizing handwritten digits

## Project structure
- `app.py` — the Streamlit dashboard and front-end experience
- `gan_mnist.py` — DCGAN model architecture, training logic, checkpointing, and generation utilities
- `digit_recognition.py` — standalone CNN architecture, upload preprocessing, and classifier loading
- `train_digit_classifier.py` — MNIST CNN training and test evaluation
- `checkpoints/mnist_cnn.pt` — trained digit-classifier weights and measured metrics
- `requirements.txt` — dependency list
- `data/` — downloaded MNIST dataset cache
- `checkpoints/` — model checkpoints saved during training
- `training_history/` — CSV history of training metrics
- `generated_samples/` — generated images and sample grids

## Run locally
1. Activate the environment:
   `./.venv/Scripts/Activate.ps1`
2. Start the app:
   `streamlit run app.py`

The upload-and-recognition page loads `checkpoints/mnist_cnn.pt` and does not train
the classifier when a user opens the app. To train or retrain it locally, run:
`python train_digit_classifier.py`

The classifier uses the MNIST test split for evaluation and saves its measured
training loss, test loss, and test accuracy in its checkpoint. Commit that
checkpoint to the same repository so Streamlit Community Cloud can load it.
The included checkpoint was trained for 3 epochs and measured a final training
loss of 0.0332, test loss of 0.0305, and test accuracy of 99.01%.

## Deploy publicly with Streamlit Community Cloud
1. Push this project to a public GitHub repository.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Create an app from the repository, select the branch, and set the main file to `app.py`.

The virtual environment, local MNIST cache, and editor settings are excluded from Git. Community Cloud may sleep when an app is inactive, and files generated while the app runs are not durable storage.

## Notes
- The app uses CPU by default and automatically switches to CUDA when available.
- Training runs to the selected total epoch target and saves a checkpoint and history after each epoch.
- Existing checkpoints are restored for generation and resumed training; the checkpoint's latent dimension is preserved.
- The latest checkpoint and generated outputs are kept on disk for recovery and reuse.
