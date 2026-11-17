import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from gan_mnist import DCGAN, ensure_directories, save_run_metadata


st.set_page_config(page_title="GAN MNIST Dashboard", page_icon="🧠", layout="wide")


def apply_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #070d1b;
            --panel: #101827;
            --panel-strong: #111b2d;
            --border: rgba(122, 151, 255, 0.2);
            --text: #e7edf7;
            --muted: #9fb6d4;
            --purple: #8b5cf6;
            --blue: #3b82f6;
            --cyan: #22d3ee;
            --success: #34d399;
            --warning: #fbbf24;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(135deg, rgba(7,13,27,0.78), rgba(9,14,26,0.82)),
                url("https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=1600&q=80") center/cover no-repeat fixed;
            color: var(--text);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(16,24,39,0.94), rgba(31,41,55,0.76));
            border: 1px solid rgba(147, 197, 253, 0.25);
            border-radius: 22px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 24px 50px rgba(15, 23, 42, 0.35);
        }
        .stSidebar {
            background: rgba(14, 20, 35, 0.86);
            border-right: 1px solid var(--border);
        }
        .card {
            background: linear-gradient(180deg, rgba(16,24,39,0.96), rgba(12,18,29,0.96));
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 18px 36px rgba(9, 12, 23, 0.35);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        h1, h2, h3, h4 {
            color: var(--text) !important;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(15,20,34,1), rgba(17,27,45,0.95));
            border: 1px solid rgba(88, 125, 255, 0.3);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem;
            min-height: 120px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.3rem;
            color: var(--text);
        }
        .status-pill {
            display: inline-block;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(34,211,238,0.12);
            color: var(--cyan);
            border: 1px solid rgba(34,211,238,0.25);
        }
        .feature-card {
            background: linear-gradient(180deg, rgba(13,19,31,0.96), rgba(20,31,45,0.9));
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1rem;
            min-height: 170px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
        }
        .feature-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.6rem;
            height: 2.6rem;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.22));
            font-size: 1.3rem;
            margin-bottom: 0.8rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, var(--purple), var(--blue));
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.62rem 1.2rem;
            box-shadow: 0 10px 20px rgba(59,130,246,0.2);
        }
        .stButton > button:hover {
            filter: brightness(1.06);
        }
        .selector .stSelectbox, .selector .stNumberInput, .selector .stSlider {
            background: rgba(11,17,28,0.8);
        }
        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.4rem;
        }
        .project-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid rgba(168, 85, 247, 0.25);
            background: rgba(139, 92, 246, 0.09);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            color: #d8c2ff;
            font-size: 0.74rem;
            font-weight: 600;
        }
        div[data-testid="stSidebarNav"] {
            padding-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    defaults = {
        "project_title": "GAN-Based Handwritten Digit Generation",
        "training_history": [],
        "epochs_completed": 0,
        "model_status": "Not trained",
        "generator_loss": None,
        "discriminator_loss": None,
        "generation_count": 0,
        "last_generated": [],
        "training_running": False,
        "current_epoch": 0,
        "total_epochs": 0,
        "training_error": "",
        "model_ready": False,
        "show_first_visit_guide": True,
        "page_navigation": "Dashboard",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault("model_config", {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.0002,
        "latent_dim": 100,
    })

    if "gan_model" not in st.session_state:
        st.session_state.gan_model = None

    history_path = Path("training_history") / "gan_history.csv"
    if history_path.exists():
        try:
            df = pd.read_csv(history_path)
            if not df.empty:
                st.session_state.training_history = df.to_dict("records")
                st.session_state.epochs_completed = max(int(df["epoch"].max()) + 1 if "epoch" in df.columns else 0, 0)
                final = df.iloc[-1]
                st.session_state.generator_loss = float(final.get("generator_loss", 0.0))
                st.session_state.discriminator_loss = float(final.get("discriminator_loss", 0.0))
        except Exception:
            pass

    checkpoint_path = DCGAN.latest_checkpoint_path("checkpoints")
    st.session_state.model_ready = checkpoint_path is not None
    if checkpoint_path:
        st.session_state.model_status = "Checkpoint Available"
    elif st.session_state.model_status != "Training Completed":
        st.session_state.model_status = "Not Trained"

    metadata_path = Path("training_history") / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("status") == "Training Completed":
            st.session_state.model_status = metadata["status"]


def make_metric_card(title, value, accent="var(--cyan)"):
    return f"""
    <div class='metric-card'>
      <div class='metric-label'>{title}</div>
      <div class='metric-value' style='color:{accent};'>{value}</div>
    </div>
    """


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_or_create_model(latent_dim: int = 100):
    if st.session_state.gan_model is None:
        device = get_device()
        checkpoint_path = DCGAN.latest_checkpoint_path("checkpoints")
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            latent_dim = int(checkpoint.get("latent_dim", latent_dim))
            model = DCGAN(latent_dim=latent_dim, device=device)
            model.load_checkpoint(checkpoint_path)
            st.session_state.model_ready = True
            st.session_state.epochs_completed = max(
                model.epochs_completed,
                max((int(item.get("epoch", -1)) + 1 for item in model.training_history), default=0),
            )
            st.session_state.training_history = model.training_history
            if model.training_history:
                latest = model.training_history[-1]
                st.session_state.generator_loss = latest.get("generator_loss")
                st.session_state.discriminator_loss = latest.get("discriminator_loss")
            if st.session_state.model_status != "Training Completed":
                st.session_state.model_status = "Checkpoint Available"
        else:
            model = DCGAN(latent_dim=latent_dim, device=device)
        st.session_state.gan_model = model
        st.session_state.model_config["device"] = device
        st.session_state.model_config["latent_dim"] = latent_dim
    return st.session_state.gan_model


def load_latest_checkpoint(model, checkpoint_dir: str = "checkpoints"):
    latest_path = DCGAN.latest_checkpoint_path(checkpoint_dir)
    if latest_path:
        model.load_checkpoint(latest_path)
        if st.session_state.model_status != "Training Completed":
            st.session_state.model_status = "Checkpoint Available"
        st.session_state.model_ready = True
        st.session_state.epochs_completed = max(
            model.epochs_completed,
            max((int(item.get("epoch", -1)) + 1 for item in model.training_history), default=0),
        )
        if model.training_history:
            last_entry = model.training_history[-1]
            st.session_state.generator_loss = last_entry.get("generator_loss")
            st.session_state.discriminator_loss = last_entry.get("discriminator_loss")
        return True
    return False


def run_training_epoch(model, dataloader, epoch_number):
    if model is None:
        return None
    record = model.train_one_epoch(dataloader, epoch_number)
    history_path = Path("training_history") / "gan_history.csv"
    model.save_training_history(str(history_path))
    checkpoint_dir = Path("checkpoints")
    model.save_checkpoint(str(checkpoint_dir), epoch_number, metadata={"epoch": epoch_number})
    st.session_state.training_history = model.training_history
    st.session_state.epochs_completed = model.epochs_completed
    st.session_state.generator_loss = record["generator_loss"]
    st.session_state.discriminator_loss = record["discriminator_loss"]
    st.session_state.model_status = "Training in progress"
    st.session_state.model_ready = True
    return record


def get_history_dataframe():
    history = st.session_state.get("training_history", [])
    if not history:
        return pd.DataFrame(columns=["epoch", "generator_loss", "discriminator_loss", "elapsed_seconds", "timestamp"])
    return pd.DataFrame(history)


def navigate_to_page(page: str):
    st.session_state.page_navigation = page


def get_dashboard_status():
    if st.session_state.get("training_running", False):
        return "Training in Progress"
    if st.session_state.get("model_status") == "Training Completed":
        return "Training Completed"
    if DCGAN.latest_checkpoint_path("checkpoints"):
        return "Checkpoint Available"
    return "Not Trained"


def render_dashboard():
    st.markdown("<div class='project-badge'>AI Research Platform • Deep Learning Demo</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='hero-panel'>
            <div style='display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;'>
                <div>
                    <h1 style='margin:0; color:#e7edf7;'>GAN-Based Handwritten Digit Generation</h1>
                    <p style='margin:0.55rem 0 0; color:#cbd5e1; max-width:900px;'>This is an interactive deep learning platform that demonstrates how Generative Adversarial Networks learn handwritten digit patterns and generate new synthetic digit images.</p>
                </div>
                <div class='status-pill'>A guided MNIST demo</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_cols = st.columns(2)
    with action_cols[0]:
        st.button(
            "Start Learning",
            key="start_learning",
            on_click=lambda: st.session_state.update(show_first_visit_guide=True),
        )
    with action_cols[1]:
        st.button("Explore Project", key="explore_project", on_click=navigate_to_page, args=("About Project",))

    history = get_history_dataframe()
    generated_count = len(list(Path("generated_samples").glob("generated_*.png")))
    checkpoint_status = "Available" if DCGAN.latest_checkpoint_path("checkpoints") else "Not available"
    status = get_dashboard_status()
    generator_loss = (
        f"{float(history.iloc[-1]['generator_loss']):.4f}"
        if not history.empty and "generator_loss" in history.columns
        else "Not available"
    )
    discriminator_loss = (
        f"{float(history.iloc[-1]['discriminator_loss']):.4f}"
        if not history.empty and "discriminator_loss" in history.columns
        else "Not available"
    )
    metric_items = [
        ("Project Status", status, "var(--success)" if status == "Training Completed" else "var(--warning)"),
        ("Model Checkpoint", checkpoint_status, "var(--cyan)"),
        ("Completed Epochs", str(st.session_state.epochs_completed), "var(--purple)"),
        ("Generated Images", str(generated_count), "var(--blue)"),
        ("Generator Loss", generator_loss, "#7dd3fc"),
        ("Discriminator Loss", discriminator_loss, "#c4b5fd"),
        ("Dataset", "MNIST (28 × 28)", "#93c5fd"),
        ("Computing Device", get_device().upper(), "#67e8f9"),
    ]
    for row_start in (0, 4):
        metric_cols = st.columns(4)
        for col, (title, value, accent) in zip(metric_cols, metric_items[row_start:row_start + 4]):
            with col:
                st.markdown(make_metric_card(title, value, accent), unsafe_allow_html=True)

    st.subheader("Getting started")
    st.caption("Follow these steps in order, or jump directly to any project section.")
    guide_controls = st.columns(2)
    with guide_controls[0]:
        st.button(
            "Skip guide",
            key="skip_guide",
            disabled=not st.session_state.show_first_visit_guide,
            on_click=lambda: st.session_state.update(show_first_visit_guide=False),
        )
    with guide_controls[1]:
        st.button(
            "Restart guide",
            key="restart_guide",
            disabled=st.session_state.show_first_visit_guide,
            on_click=lambda: st.session_state.update(show_first_visit_guide=True),
        )

    if st.session_state.show_first_visit_guide:
        guide_steps = [
            ("1 · Understand the project", "This app teaches how a GAN (Generative Adversarial Network) learns to make new digit images. MNIST is a collection of real handwritten digits from 0 to 9. You will explore the data, see the two-part model, train it, and inspect its output.", "About Project", "Learn the basics"),
            ("2 · Explore the dataset", "MNIST contains 60,000 training images. Each is a small 28 × 28 grayscale picture with a real digit label.", "Dataset Explorer", "Explore Dataset"),
            ("3 · Understand the model", "The Generator makes artificial digits. The Discriminator examines an image and learns whether it came from MNIST or the Generator.", "Model Architecture", "View Architecture"),
            ("4 · Train the model", "Training means repeating examples so both networks improve. An epoch is one pass through the training set; batch size is how many examples are processed together; learning rate controls how large each learning update is. Training only starts when you choose Start training.", "Train Model", "Train Model"),
            ("5 · Generate digits", "Choose how many images to create, then press Generate. The Generator starts from random noise—a list of numbers—and turns it into digit-like pixels.", "Generate Digits", "Generate Digits"),
            ("6 · Read training analytics", "Generator loss measures how difficult it is for the Generator to fool the Discriminator. Discriminator loss measures its mistakes on real and generated images. Losses are training signals, not image-quality scores by themselves.", "Training Analytics", "View Analytics"),
            ("7 · View the image gallery", "The gallery shows PNG images saved by the generation page. Use its download control to keep generated results.", "Image Gallery", "Open Gallery"),
        ]
        for title, description, page, button_label in guide_steps:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(description)
                st.button(button_label, key=f"guide_{page}", on_click=navigate_to_page, args=(page,))

    st.subheader("How the project works")
    workflow = [
        ("MNIST Dataset", "Real labeled digit pictures"),
        ("Data Preprocessing", "Convert pixels to tensors and normalize them for training"),
        ("Generator + Discriminator", "One creates digits; the other learns to distinguish them"),
        ("Adversarial Training", "Both networks improve through repeated feedback"),
        ("Trained Generator", "The Generator learns patterns from the training process"),
        ("Synthetic Digits", "New images are made from random noise"),
    ]
    for row_start in (0, 3):
        workflow_cols = st.columns(3)
        for col, (index, (title, detail)) in zip(
            workflow_cols,
            enumerate(workflow[row_start:row_start + 3], start=row_start),
        ):
            with col:
                st.markdown(
                    f"<div class='feature-card'><div class='feature-icon'>{index + 1}</div>"
                    f"<h4 style='margin:0 0 0.5rem;color:#f8fafc;'>{title}</h4>"
                    f"<p style='margin:0;color:#cbd5e1;line-height:1.5;'>{detail}</p></div>",
                    unsafe_allow_html=True,
                )
        if row_start == 0:
            st.markdown("<div style='text-align:center;color:#22d3ee;font-size:1.4rem;'>↓</div>", unsafe_allow_html=True)

    with st.expander("Beginner glossary"):
        glossary = [
            ("Generator", "The network that creates a new digit picture."),
            ("Discriminator", "The network that tries to tell real MNIST images from generated ones."),
            ("Epoch", "One full pass through all training examples."),
            ("Random noise", "A random numeric starting point from which the Generator creates an image."),
            ("Generator loss", "A training signal showing how well the Generator is currently fooling the Discriminator."),
            ("Discriminator loss", "A training signal showing how well the Discriminator currently identifies real and generated images."),
            ("Checkpoint", "A saved snapshot of model weights that can be loaded later; it does not by itself mean training is finished."),
            ("Synthetic image", "A computer-generated image rather than a handwritten image from the dataset."),
        ]
        for term, explanation in glossary:
            st.markdown(f"**{term}:** {explanation}")

    st.subheader("Explore the project")
    nav_cards = [
        ("Dashboard", "Project overview and current status.", "Open Dashboard"),
        ("Dataset Explorer", "Explore real MNIST samples and their labels.", "Open Dataset"),
        ("Model Architecture", "See the Generator and Discriminator diagrams.", "View Architecture"),
        ("Train Model", "Choose training settings and manually start training.", "Open Training"),
        ("Generate Digits", "Select an image count and generate samples.", "Open Generator"),
        ("Training Analytics", "Review measured losses and training time.", "View Analytics"),
        ("Image Gallery", "Browse and download saved generated images.", "Open Gallery"),
        ("About Project", "Read the project objective and technologies.", "About Project"),
    ]
    for row_start in range(0, len(nav_cards), 4):
        nav_cols = st.columns(4)
        for col, (page, description, button_label) in zip(nav_cols, nav_cards[row_start:row_start + 4]):
            with col:
                st.markdown(
                    f"<div class='feature-card'><h4 style='margin:0 0 0.5rem;color:#f8fafc;'>{page}</h4>"
                    f"<p style='margin:0 0 0.8rem;color:#cbd5e1;line-height:1.5;'>{description}</p></div>",
                    unsafe_allow_html=True,
                )
                st.button(button_label, key=f"nav_{page}", on_click=navigate_to_page, args=(page,))


def render_model_architecture():
    st.title("Model Architecture")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Generator architecture")
    st.write("Random Noise → Dense Layer → Reshape → Transposed Convolution → Batch Normalization → ReLU → Tanh → Generated Image")
    st.markdown("</div>", unsafe_allow_html=True)

    generator_layers = [
        ("Noise", "100-D vector", "Random latent vector"),
        ("Dense", "(100 → 6272)", "Expands latent vector to feature map"),
        ("Reshape", "(128 × 7 × 7)", "Changes signal to image-like structure"),
        ("ConvTranspose 1", "(128 → 64)", "Upsamples to 14 × 14"),
        ("BatchNorm", "64 channels", "Stabilizes feature distribution"),
        ("ConvTranspose 2", "(64 → 1)", "Upsamples to 28 × 28 grayscale image"),
        ("Tanh", "28 × 28", "Produces pixel values in [-1, 1]"),
    ]

    fig_generator = go.Figure()
    x_positions = [0.9, 2.5, 4.2, 6.0, 7.8, 9.6, 11.3]
    for x, (name, dims, purpose) in zip(x_positions, generator_layers):
        fig_generator.add_shape(type="rect", x0=x - 0.7, y0=0.08, x1=x + 0.7, y1=1.25, line=dict(color="#8b5cf6", width=2), fillcolor="rgba(139,92,246,0.18)")
        fig_generator.add_annotation(x=x, y=0.68, text=f"{name}<br>{dims}", showarrow=False, font=dict(size=12, color="#f5f3ff"))
        fig_generator.add_annotation(x=x, y=1.5, text=purpose, showarrow=False, font=dict(size=9, color="#cbd5e1"))
    fig_generator.update_layout(
        width=1200,
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(range=[-0.2, 12.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.3, 1.9], showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig_generator, use_container_width=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Discriminator architecture")
    st.write("Input Image → Convolution Layers → LeakyReLU → Flatten → Fully Connected Layer → Sigmoid → Real/Fake Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

    discriminator_layers = [
        ("Input", "28 × 28 × 1", "Real or generated grayscale image"),
        ("Conv 1", "(1 → 64)", "Extracts low-level edges"),
        ("LeakyReLU", "64 channels", "Allows gradients for negative activations"),
        ("Conv 2", "(64 → 128)", "Captures higher-level patterns"),
        ("Flatten", "128 × 7 × 7", "Reshapes feature map to vector"),
        ("Linear", "(6272 → 1)", "Scores authenticity of the image"),
        ("Sigmoid", "0 to 1", "Classifies real vs fake"),
    ]

    fig_discriminator = go.Figure()
    x_positions = [0.9, 2.5, 4.2, 6.0, 7.8, 9.6, 11.3]
    for x, (name, dims, purpose) in zip(x_positions, discriminator_layers):
        fig_discriminator.add_shape(type="rect", x0=x - 0.7, y0=0.08, x1=x + 0.7, y1=1.25, line=dict(color="#10b981", width=2), fillcolor="rgba(16,185,129,0.16)")
        fig_discriminator.add_annotation(x=x, y=0.68, text=f"{name}<br>{dims}", showarrow=False, font=dict(size=12, color="#ecfeff"))
        fig_discriminator.add_annotation(x=x, y=1.5, text=purpose, showarrow=False, font=dict(size=9, color="#cbd5e1"))
    fig_discriminator.update_layout(
        width=1200,
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(range=[-0.2, 12.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.3, 1.9], showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig_discriminator, use_container_width=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Adversarial training behavior")
    st.write(
        "The generator tries to fool the discriminator by creating realistic digits, while the discriminator improves its ability to detect synthetic images. This competitive training pushes both networks toward equilibrium and produces increasingly authentic digit samples."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_training_page():
    st.title("Train Model")
    st.caption("Train the DCGAN on MNIST. The model trains on CPU by default and switches to CUDA automatically when available.")

    if st.session_state.gan_model is None and DCGAN.latest_checkpoint_path("checkpoints"):
        load_or_create_model(latent_dim=st.session_state.model_config.get("latent_dim", 100))

    config = st.session_state.model_config.copy()
    checkpoint_available = DCGAN.latest_checkpoint_path("checkpoints") is not None
    with st.form("training_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            config["epochs"] = st.number_input("Total epochs target", min_value=1, max_value=200, value=int(config.get("epochs", 5)))
        with col2:
            config["batch_size"] = st.number_input("Batch size", min_value=16, max_value=512, value=int(config.get("batch_size", 64)))
        with col3:
            config["learning_rate"] = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=float(config.get("learning_rate", 0.0002)), step=1e-5, format="%.5f")

        col4, col5 = st.columns(2)
        with col4:
            config["latent_dim"] = st.number_input(
                "Latent dimension",
                min_value=10,
                max_value=500,
                value=int(config.get("latent_dim", 100)),
                disabled=checkpoint_available,
                help="A saved checkpoint fixes this value so its Generator architecture can be resumed safely." if checkpoint_available else None,
            )
        with col5:
            st.write("Training device")
            st.write(f"{get_device().upper()} available")

        train_cols = st.columns(2)
        with train_cols[0]:
            start_training = st.form_submit_button("Start training")
        with train_cols[1]:
            load_training = st.form_submit_button("Load checkpoint")

    st.session_state.model_config = config

    if load_training:
        model = load_or_create_model(latent_dim=config["latent_dim"])
        if load_latest_checkpoint(model, checkpoint_dir="checkpoints"):
            st.success("Latest checkpoint loaded.")
        else:
            st.info("No saved checkpoint is available yet.")

    if start_training and int(config["epochs"]) <= st.session_state.epochs_completed:
        st.info(
            f"The model already has {st.session_state.epochs_completed} completed epochs. "
            "Increase the total epoch target to continue training."
        )
        start_training = False

    if start_training:
        st.session_state.training_running = True
        st.session_state.current_epoch = st.session_state.epochs_completed
        st.session_state.total_epochs = int(config["epochs"])
        st.session_state.model_status = "Training in Progress"
        st.session_state.training_error = ""
        save_run_metadata(".", {**config, "status": "Training in Progress"})

        model = load_or_create_model(latent_dim=config["latent_dim"])
        if model.latent_dim != int(config["latent_dim"]):
            st.session_state.training_running = False
            st.error(
                f"The saved checkpoint uses latent dimension {model.latent_dim}. "
                "The saved model must be resumed with that same dimension."
            )
            return
        for optimizer in (model.generator_optimizer, model.discriminator_optimizer):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = float(config["learning_rate"])
        device = get_device()
        model.device = device
        model.generator.to(device)
        model.discriminator.to(device)

        try:
            dataloader = model.get_data_loader(batch_size=config["batch_size"], root_dir="./data")
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_box = st.empty()

            start_epoch = st.session_state.epochs_completed
            target_epoch = int(config["epochs"])
            for epoch_number in range(st.session_state.epochs_completed, target_epoch):
                record = model.train_one_epoch(dataloader, epoch_number)
                history_path = Path("training_history") / "gan_history.csv"
                model.save_training_history(str(history_path))
                checkpoint_dir = Path("checkpoints")
                model.save_checkpoint(str(checkpoint_dir), epoch_number, metadata={"epoch": epoch_number})

                st.session_state.training_history = model.training_history
                st.session_state.epochs_completed = model.epochs_completed
                st.session_state.generator_loss = record["generator_loss"]
                st.session_state.discriminator_loss = record["discriminator_loss"]
                st.session_state.model_status = "Training in progress"
                st.session_state.model_ready = True

                progress_value = (epoch_number + 1 - start_epoch) / max(target_epoch - start_epoch, 1)
                progress_bar.progress(progress_value)
                status_text.write(f"Epoch {epoch_number + 1} / {target_epoch} complete")
                metrics_box.write(f"Generator loss: {record['generator_loss']:.4f} | Discriminator loss: {record['discriminator_loss']:.4f}")

            if st.session_state.training_running:
                st.session_state.training_running = False
                st.session_state.model_status = "Training Completed"
                save_run_metadata(".", {**config, "status": "Training Completed"})
                st.success("Training completed successfully.")
        except Exception as exc:
            st.session_state.training_error = str(exc)
            st.session_state.training_running = False
            st.session_state.model_status = "Checkpoint Available" if DCGAN.latest_checkpoint_path("checkpoints") else "Not Trained"
            save_run_metadata(".", {**config, "status": st.session_state.model_status})
            st.error(f"Training failed: {exc}")

    if st.session_state.training_error:
        st.error(st.session_state.training_error)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current training state")
    if st.session_state.training_running:
        st.write("Status: Running")
    else:
        st.write("Status: Ready")
    st.write(f"Completed epochs: {st.session_state.epochs_completed}")
    st.write(f"Latest generator loss: {st.session_state.generator_loss if st.session_state.generator_loss is not None else 'Not available'}")
    st.write(f"Latest discriminator loss: {st.session_state.discriminator_loss if st.session_state.discriminator_loss is not None else 'Not available'}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_generate_page():
    st.title("Generate Digits")

    model = st.session_state.gan_model
    if model is None:
        model = load_or_create_model(latent_dim=st.session_state.model_config.get("latent_dim", 100))

    if st.session_state.model_ready is False and not model.training_history:
        st.warning("A trained model is required before generating digits. Train the DCGAN first or load a saved checkpoint.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        count = st.number_input("Number of images", min_value=1, max_value=25, value=9)
    with col2:
        seed = st.number_input("Fixed seed (optional)", min_value=-1, max_value=9999, value=-1, step=1)
    with col3:
        preset = st.selectbox("Preset grid", [1, 4, 9, 16, 25], index=2)
        if st.button("Use preset"):
            count = preset

    if st.button("Generate"):
        if seed >= 0:
            seed_value = int(seed)
        else:
            seed_value = None

        model = load_or_create_model(latent_dim=st.session_state.model_config.get("latent_dim", 100))
        output_dir = Path("generated_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_paths = model.generate_images(count=count, seed=seed_value, base_dir=str(output_dir))
        st.session_state.last_generated = generated_paths
        st.session_state.generation_count = count

        images = [Image.open(path) for path in generated_paths]
        st.subheader("Generated images")
        grid_cols = st.columns(5)
        for idx, img in enumerate(images):
            with grid_cols[idx % 5]:
                st.image(img, caption=f"Digit {idx + 1}", use_container_width=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Downloads")
        fig_path = model.save_generated_grid(str(output_dir), count=min(count, 25), filename="generated_grid.png", seed=seed_value)
        st.download_button("Download generated grid", data=fig_path.read_bytes(), file_name="generated_grid.png", mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            from torchvision import datasets, transforms
            real_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
            indices = [i % len(real_dataset) for i in range(min(count, 9))]
            st.subheader("Real vs generated comparison")
            compare_cols = st.columns(min(count, 9))
            for idx, dataset_index in enumerate(indices):
                image, label = real_dataset[dataset_index]
                with compare_cols[idx]:
                    st.image(image.squeeze(0).numpy(), caption=f"Real: {label}", use_container_width=True)
                    st.image(Image.open(generated_paths[idx]), caption=f"Generated {idx + 1}", use_container_width=True)
        except Exception as exc:
            st.caption(f"Real-image comparison preview unavailable: {exc}")
    elif st.session_state.last_generated:
        images = [Image.open(path) for path in st.session_state.last_generated]
        grid_cols = st.columns(5)
        for idx, img in enumerate(images):
            with grid_cols[idx % 5]:
                st.image(img, caption=f"Digit {idx + 1}", use_container_width=True)


def render_analytics_page():
    st.title("Training Analytics")
    history = get_history_dataframe()

    if history.empty:
        st.info("No training history available yet. Train the model to begin collecting analytics.")
        return

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=history["epoch"], y=history["generator_loss"], mode="lines+markers", name="Generator Loss", line=dict(color="#8b5cf6", width=3)))
    fig_loss.add_trace(go.Scatter(x=history["epoch"], y=history["discriminator_loss"], mode="lines+markers", name="Discriminator Loss", line=dict(color="#22d3ee", width=3)))
    fig_loss.update_layout(title="Loss vs Epoch", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_loss, use_container_width=True)

    fig_progress = go.Figure()
    fig_progress.add_trace(go.Scatter(x=history["epoch"], y=history["elapsed_seconds"], mode="lines+markers", name="Training time", line=dict(color="#60a5fa", width=3)))
    fig_progress.update_layout(title="Training Duration", xaxis_title="Epoch", yaxis_title="Seconds", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_progress, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance summary")
        last = history.iloc[-1]
        st.write(f"Final generator loss: {last['generator_loss']:.4f}")
        st.write(f"Final discriminator loss: {last['discriminator_loss']:.4f}")
        st.write(f"Total training time: {history['elapsed_seconds'].sum():.2f} seconds")
    with col2:
        st.subheader("Observations")
        st.write("- The generator loss and discriminator loss are tracked after each epoch.")
        st.write("- A good DCGAN training run tends to reduce both losses together as the model learns.")
        st.write("- Generated image quality can improve significantly with additional training epochs.")


def render_gallery_page():
    st.title("Image Gallery")
    samples_dir = Path("generated_samples")
    if not samples_dir.exists() or not list(samples_dir.glob("*.png")):
        st.info("No generated images saved yet. Generate one or more images from the Generate Digits page.")
        return

    image_files = sorted(samples_dir.glob("*.png"))
    for row_start in range(0, len(image_files), 4):
        image_cols = st.columns(4)
        for col, image_path in zip(image_cols, image_files[row_start:row_start + 4]):
            with col:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)
                st.download_button(
                    f"Download {image_path.name}",
                    data=image_path.read_bytes(),
                    file_name=image_path.name,
                    mime="image/png",
                    key=f"download_{image_path.name}",
                )


def render_dataset_page():
    st.title("Dataset Explorer")
    st.caption("Exploring the MNIST dataset used to train the DCGAN.")

    dataset_dir = "./data"
    try:
        from torchvision import datasets, transforms

        dataset = datasets.MNIST(
            root=dataset_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        counts = {str(i): int((np.array(dataset.targets) == i).sum()) for i in range(10)}

        sample_count = st.slider("Number of samples", min_value=1, max_value=min(20, len(dataset)), value=min(10, len(dataset)))
        sample_indices = list(range(sample_count))

        st.write(f"Total dataset examples: {len(dataset)}")
        st.write(f"Image size: {tuple(dataset[0][0].shape[-2:])}")

        col1, col2 = st.columns([1.5, 1.2])
        with col1:
            sample_cols = st.columns(min(5, sample_count))
            for index, dataset_index in enumerate(sample_indices):
                image, label = dataset[dataset_index]
                image_array = image.squeeze(0).numpy() if image.ndim == 3 else image.numpy()
                with sample_cols[index % len(sample_cols)]:
                    st.image(image_array, caption=f"Digit {label}", clamp=True, use_container_width=True)
        with col2:
            st.subheader("Class distribution")
            fig = go.Figure(go.Bar(
                x=[str(k) for k in range(10)],
                y=[counts[str(i)] for i in range(10)],
                marker=dict(color="#8b5cf6", line=dict(color="#c4b5fd", width=1.2)),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Dataset could not be loaded: {exc}")


def render_about_page():
    st.title("About Project")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("This project demonstrates a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST handwritten digit dataset.")
    st.write("The goal is to create realistic synthetic digits by learning the underlying distribution of handwritten characters.")
    st.write("It is designed for a college-level deep learning presentation and uses PyTorch for the neural network, Streamlit for the dashboard, and Plotly for analytics.")
    st.markdown("</div>", unsafe_allow_html=True)


def app_main():
    apply_custom_styles()
    initialize_session_state()

    sidebar = st.sidebar
    sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
    page = sidebar.radio("Navigation page", [
        "Dashboard",
        "Dataset Explorer",
        "Model Architecture",
        "Train Model",
        "Generate Digits",
        "Training Analytics",
        "Image Gallery",
        "About Project",
    ], key="page_navigation", label_visibility="collapsed")

    pages = {
        "Dashboard": render_dashboard,
        "Model Architecture": render_model_architecture,
        "Train Model": render_training_page,
        "Generate Digits": render_generate_page,
        "Training Analytics": render_analytics_page,
        "Image Gallery": render_gallery_page,
        "Dataset Explorer": render_dataset_page,
        "About Project": render_about_page,
    }

    pages.get(page, render_dashboard)()


if __name__ == "__main__":
    ensure_directories(".")
    app_main()
