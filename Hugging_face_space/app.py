import spaces
import torch
import gradio as gr

from torchvision import transforms
from huggingface_hub import hf_hub_download

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# Hugging Face Model Configuration
# ============================================================

REPO_ID = "Ayush5605/NST_Model"


# ============================================================
# Device
# ============================================================

device = torch.device("cuda")

print("=" * 60)
print(f"Using device: {device}")
print("=" * 60)


# ============================================================
# Download Model Files from Hugging Face
# ============================================================

print("Downloading/loading VGG model...")

vgg_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="vgg_normalised.pth"
)

print("VGG model downloaded successfully.")

print("Downloading/loading decoder model...")

decoder_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="decoder_80.pth"
)

print("Decoder model downloaded successfully.")


# ============================================================
# Load VGG Encoder
# ============================================================

print("Loading VGG Encoder...")

encoder = VGGEncoder(vgg_path).to(device)


# ============================================================
# Load Decoder
# ============================================================

print("Loading Decoder...")

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        decoder_path,
        map_location=device
    )
)


# ============================================================
# Evaluation Mode
# ============================================================

encoder.eval()
decoder.eval()

print("Models loaded successfully!")


# ============================================================
# Neural Style Transfer
# ============================================================

def style_transfer(
    content_image,
    style_image,
    alpha
):
    """
    Perform neural style transfer using:

    Content Image
        ↓
    VGG Encoder
        ↓
    Content Features

    Style Image
        ↓
    VGG Encoder
        ↓
    Style Features

    Content + Style Features
        ↓
    AdaIN
        ↓
    Alpha Blending
        ↓
    Decoder
        ↓
    Stylized Image
    """

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    # --------------------------------------------------------
    # Convert images to tensors
    # --------------------------------------------------------

    content_tensor = content_transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = style_transform(
        style_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Keep alpha in valid range
    # --------------------------------------------------------

    alpha = max(
        0.0,
        min(1.0, float(alpha))
    )

    # --------------------------------------------------------
    # Perform inference
    # --------------------------------------------------------

    with torch.inference_mode():

        # Extract content features
        content_feats = encoder(
            content_tensor,
            is_test=True
        )

        # Extract style features
        style_feats = encoder(
            style_tensor,
            is_test=True
        )

        # ----------------------------------------------------
        # Adaptive Instance Normalization
        # ----------------------------------------------------

        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        # ----------------------------------------------------
        # Alpha blending
        # ----------------------------------------------------

        stylized_feats = (
            alpha * stylized_feats
            +
            (1 - alpha) * content_feats
        )

        # ----------------------------------------------------
        # Decode stylized features
        # ----------------------------------------------------

        stylized_tensor = decoder(
            stylized_feats
        )

    # ========================================================
    # Convert tensor -> PIL image
    # ========================================================

    stylized_tensor = stylized_tensor.cpu().clone()

    stylized_tensor = stylized_tensor.squeeze(0)

    stylized_tensor = stylized_tensor.clamp(0, 1)

    result = transforms.ToPILImage()(
        stylized_tensor
    )

    return result


# ============================================================
# ZeroGPU Function
# ============================================================

@spaces.GPU(duration=60)
def generate(
    content_image,
    style_image,
    alpha
):
    """
    Generate a stylized image from
    content and style images.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if content_image is None:
        raise gr.Error(
            "Please upload a content image."
        )

    if style_image is None:
        raise gr.Error(
            "Please upload a style image."
        )

    try:

        # ----------------------------------------------------
        # Ensure RGB images
        # ----------------------------------------------------

        content_image = content_image.convert("RGB")

        style_image = style_image.convert("RGB")

        # ----------------------------------------------------
        # Run style transfer
        # ----------------------------------------------------

        return style_transfer(
            content_image,
            style_image,
            alpha
        )

    except Exception as e:

        print(
            f"Style transfer error: {e}"
        )

        raise gr.Error(
            f"Style transfer failed: {str(e)}"
        )


# ============================================================
# Custom CSS
# ============================================================

custom_css = """

:root {
    --radius-lg: 18px;
    --radius-md: 14px;
}


/* Main container */

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}


/* Hero section */

.hero {
    text-align: center;
    padding: 28px 20px 18px;
}


.hero h1 {
    font-size: 42px !important;
    margin-bottom: 8px !important;
    letter-spacing: -1px;
}


.hero p {
    font-size: 17px;
    opacity: 0.82;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
}


/* Cards */

.section-card {
    border-radius: var(--radius-lg) !important;
    padding: 18px !important;
}


/* Section titles */

.step-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}


/* Generate button */

.generate-btn {
    min-height: 52px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
}


/* Alpha information */

.alpha-info {
    text-align: center;
    opacity: 0.72;
    font-size: 13px;
}


/* Footer */

.footer {
    text-align: center;
    opacity: 0.65;
    font-size: 13px;
    padding: 12px 0 24px;
}

"""


# ============================================================
# Reset Function
# ============================================================

def reset_inputs():
    return (
        None,
        None,
        1.0,
        None
    )


# ============================================================
# Gradio Interface
# ============================================================

with gr.Blocks(
    title="Neural Style Transfer | AdaIN"
) as demo:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <h1>🎨 Neural Style Transfer</h1>

            <p>
                Transform your photos using the artistic style
                of another image with
                <b>AdaIN (Adaptive Instance Normalization)</b>.
            </p>

        </div>
        """
    )


    # ========================================================
    # MAIN WORKSPACE
    # ========================================================

    with gr.Row(equal_height=False):

        # ====================================================
        # LEFT: INPUTS
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes="section-card"
        ):

            gr.Markdown(
                "### 🖼️ 1. Choose your images"
            )


            # ------------------------------------------------
            # Content image
            # ------------------------------------------------

            content_input = gr.Image(
                label="Content Image",
                type="pil",
                sources=["upload"],
                height=280
            )


            # ------------------------------------------------
            # Style image
            # ------------------------------------------------

            style_input = gr.Image(
                label="Style Image",
                type="pil",
                sources=["upload"],
                height=280
            )


            # ------------------------------------------------
            # Style strength
            # ------------------------------------------------

            gr.Markdown(
                "### 🎚️ 2. Adjust style strength"
            )


            alpha_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Style Strength",
                info=(
                    "0 = original content • "
                    "1 = maximum style"
                )
            )


            gr.Markdown(
                """
                <div class='alpha-info'>
                    Higher values apply more of the artistic style.
                </div>
                """
            )


            # ------------------------------------------------
            # Buttons
            # ------------------------------------------------

            with gr.Row():

                transfer_button = gr.Button(
                    "✨ Generate Stylized Image",
                    variant="primary",
                    elem_classes="generate-btn",
                    scale=3
                )


                clear_button = gr.Button(
                    "↺ Clear",
                    variant="secondary",
                    scale=1
                )


        # ====================================================
        # RIGHT: OUTPUT
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes="section-card"
        ):

            gr.Markdown(
                "### ✨ 3. Your result"
            )


            # ------------------------------------------------
            # IMPORTANT:
            # show_download_button was removed because
            # Gradio 6 does not support it.
            # ------------------------------------------------

            output_image = gr.Image(
                label="Stylized Image",
                type="pil",
                height=600
            )


            gr.Markdown(
                """
                **Tip:** For the best results, use clear images
                with reasonably similar composition or subject
                placement.
                """
            )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    with gr.Accordion(
        "🧠 How does it work?",
        open=False
    ):

        gr.Markdown(
            """
            ### AdaIN Pipeline

            **Content Image**
            → VGG Encoder
            → Content Features

            **Style Image**
            → VGG Encoder
            → Style Features

            **AdaIN**
            → Aligns feature statistics of content and style

            **Alpha Blending**
            → Controls the strength of the transferred style

            **Decoder**
            → Reconstructs the final stylized image

            The application uses a pretrained
            **VGG encoder** and the trained **decoder model**
            hosted on Hugging Face.
            """
        )


    # ========================================================
    # QUICK GUIDE
    # ========================================================

    with gr.Accordion(
        "🚀 Quick guide",
        open=False
    ):

        gr.Markdown(
            """
            **1. Upload a content image**

            This is the image whose structure you want to preserve.

            **2. Upload a style image**

            This provides the artistic appearance.

            **3. Set Style Strength**

            Start around **0.7–1.0** and adjust according to
            the result.

            **4. Generate**

            Click **Generate Stylized Image** and wait for the
            ZeroGPU inference to finish.

            **5. Download**

            Use the download button on the generated image.
            """
        )


    # ========================================================
    # EVENTS
    # ========================================================

    transfer_button.click(
        fn=generate,
        inputs=[
            content_input,
            style_input,
            alpha_input
        ],
        outputs=output_image
    )


    clear_button.click(
        fn=reset_inputs,
        inputs=[],
        outputs=[
            content_input,
            style_input,
            alpha_input,
            output_image
        ]
    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">
            Built with PyTorch • VGG • AdaIN • Gradio •
            Hugging Face ZeroGPU
        </div>
        """
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate"
        ),
        css=custom_css
    )