from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

DATASETS = [
    Path(r"D:\Projects\NST_dataset\Content_dataset\train"),
    Path(r"D:\Projects\NST_dataset\Content_dataset\test"),
    Path(r"D:\Projects\NST_dataset\Style_dataset\train"),
    Path(r"D:\Projects\NST_dataset\Style_dataset\test"),
]

# Maximum width/height of an image after preprocessing
MAX_SIZE = 2048

# JPEG quality
JPEG_QUALITY = 95


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_path):

    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        with Image.open(image_path) as img:

            # Convert to RGB
            img = img.convert("RGB")

            original_width, original_height = img.size

            # ------------------------------------------------
            # Check whether resizing is necessary
            # ------------------------------------------------

            if (
                original_width <= MAX_SIZE
                and original_height <= MAX_SIZE
            ):
                return "unchanged"

            # ------------------------------------------------
            # Resize while maintaining aspect ratio
            # ------------------------------------------------

            img.thumbnail(
                (MAX_SIZE, MAX_SIZE),
                Image.Resampling.LANCZOS
            )

            # ------------------------------------------------
            # Save back to same file
            # ------------------------------------------------

            img.save(
                image_path,
                "JPEG",
                quality=JPEG_QUALITY,
                optimize=True
            )

            return (
                f"resized "
                f"{original_width}x{original_height} "
                f"-> {img.width}x{img.height}"
            )

    except Exception as e:

        return f"ERROR: {e}"


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(dataset_path):

    print("\n" + "=" * 70)
    print(f"Processing: {dataset_path}")
    print("=" * 70)

    if not dataset_path.exists():

        print("Dataset does not exist!")
        return

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tiff",
        ".tif"
    }

    images = [
        p
        for p in dataset_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(images)} images.")

    resized = 0
    unchanged = 0
    errors = 0

    for image_path in tqdm(images):

        result = process_image(image_path)

        if result == "unchanged":

            unchanged += 1

        elif result.startswith("resized"):

            resized += 1

            print(
                f"\n{image_path.name}: {result}"
            )

        elif result.startswith("ERROR"):

            errors += 1

            print(
                f"\n{image_path.name}: {result}"
            )

    print("\nResults:")
    print(f"Unchanged : {unchanged}")
    print(f"Resized   : {resized}")
    print(f"Errors    : {errors}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEURAL STYLE TRANSFER DATASET PREPROCESSING")
    print("=" * 70)

    print(
        f"\nMaximum image dimension: {MAX_SIZE}x{MAX_SIZE}"
    )

    print(
        "\nThe script will resize very large images "
        "while preserving aspect ratio."
    )

    for dataset in DATASETS:

        process_dataset(dataset)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()