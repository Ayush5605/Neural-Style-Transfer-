import os

import torch

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap

from werkzeug.utils import secure_filename

from wtforms import (
    FileField,
    SubmitField,
    FloatField,
    HiddenField
)

from PIL import Image
from torchvision import transforms

from huggingface_hub import hf_hub_download

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# Flask Configuration
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key"
)

app.config["UPLOAD_FOLDER"] = os.path.join(
    app.root_path,
    "static",
    "uploads"
)

app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg"
}

Bootstrap(app)

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# Upload Form
# ============================================================

class UploadForm(FlaskForm):

    content = FileField("Content Image")

    style = FileField("Style Image")

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField("Transfer Style")


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

print("=" * 60)


# ============================================================
# Hugging Face Model Configuration
# ============================================================

REPO_ID = "Ayush5605/NST_Model"


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

encoder = VGGEncoder(
    vgg_path
).to(device)


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
# Utility Functions
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# ============================================================
# Neural Style Transfer
# ============================================================

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    # --------------------------------------------------------
    # Image transformations
    # --------------------------------------------------------

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

    content_image = content_transform(
        content_image
    ).unsqueeze(0).to(device)

    style_image = style_transform(
        style_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Perform inference
    # --------------------------------------------------------

    with torch.no_grad():

        # Extract content features
        content_feats = encoder(
            content_image,
            is_test=True
        )

        # Extract style features
        style_feats = encoder(
            style_image,
            is_test=True
        )

        # Adaptive Instance Normalization
        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        # Apply alpha
        stylized_feats = (
            alpha * stylized_feats
            +
            (1 - alpha) * content_feats
        )

        # Decode stylized features
        stylized_image = decoder(
            stylized_feats
        )

    return stylized_image


# ============================================================
# Save Generated Image
# ============================================================

def save_image(image, path):

    image = image.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)


# ============================================================
# Main Route
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        # ====================================================
        # Content Image
        # ====================================================

        if (
            form.content.data
            and
            form.content.data.filename
        ):

            if allowed_file(
                form.content.data.filename
            ):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                form.content.data.save(
                    os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        content_filename
                    )
                )

                form.content_path.data = content_filename

        else:

            content_filename = form.content_path.data


        # ====================================================
        # Style Image
        # ====================================================

        if (
            form.style.data
            and
            form.style.data.filename
        ):

            if allowed_file(
                form.style.data.filename
            ):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                form.style.data.save(
                    os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        style_filename
                    )
                )

                form.style_path.data = style_filename

        else:

            style_filename = form.style_path.data


        # ====================================================
        # Perform Style Transfer
        # ====================================================

        if (
            content_filename
            and
            style_filename
        ):

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                content_filename
            )

            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                style_filename
            )

            try:

                # Open images
                content_image = Image.open(
                    content_path
                ).convert("RGB")

                style_image = Image.open(
                    style_path
                ).convert("RGB")


                # Get alpha value
                alpha = float(
                    form.alpha.data
                )

                # Keep alpha in valid range
                alpha = max(
                    0.0,
                    min(1.0, alpha)
                )


                # Perform NST
                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )


                # Output filename
                result_filename = (
                    "stylized_"
                    +
                    content_filename
                )


                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )


                # Save result
                save_image(
                    stylized_image,
                    result_path
                )


                result_image = result_filename


            except Exception as e:

                print(
                    f"Style transfer error: {e}"
                )

                error = str(e)

    else:

        # Show errors when POST was attempted
        if request.method == "POST":

            if not content_filename:

                error = "Please upload content image"

            elif not style_filename:

                error = "Please upload style image"


    # ========================================================
    # Render HTML
    # ========================================================

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# Serve Uploaded Images
# ============================================================

@app.route("/uploads/<filename>")
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# Serve Example Images
# ============================================================

@app.route("/examples/<path:filename>")
def send_example(filename):

    return send_from_directory(
        os.path.join(
            app.root_path,
            "examples"
        ),
        filename
    )


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                10000
            )
        ),
        debug=False
    )