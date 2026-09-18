from pathlib import Path

from PIL import Image, ImageFile
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

# Maximum width/height after preprocessing
MAX_SIZE = 2048

# JPEG quality
JPEG_QUALITY = 95

# Allow PIL to detect/load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_path):

    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        with Image.open(image_path) as img:

            # Force PIL to actually read the complete image.
            # This helps detect truncated/corrupted files.
            img.load()

            # ------------------------------------------------
            # Convert to RGB
            # ------------------------------------------------

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
            # Save resized image
            # ------------------------------------------------

            # If the original is already JPEG, overwrite it.
            # Otherwise create a JPEG version and remove the
            # old file.

            if image_path.suffix.lower() in [".jpg", ".jpeg"]:

                img.save(
                    image_path,
                    "JPEG",
                    quality=JPEG_QUALITY,
                    optimize=True
                )

            else:

                new_path = image_path.with_suffix(".jpg")

                img.save(
                    new_path,
                    "JPEG",
                    quality=JPEG_QUALITY,
                    optimize=True
                )

                # Remove original file
                image_path.unlink()

            return (
                f"resized "
                f"{original_width}x{original_height} "
                f"-> {img.width}x{img.height}"
            )

    except Exception as e:

        print(f"\n❌ Corrupted image: {image_path.name}")
        print(f"   Error: {e}")

        try:
             image_path.unlink()
             print("   🗑️ Image removed.")
        except Exception as delete_error:
              print(f"   ⚠️ Could not remove image: {delete_error}")

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

    corrupted_files = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

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

            corrupted_files.append(
                (image_path, result)
            )

            print(
                f"\n❌ {image_path.name}: {result}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nResults:")
    print(f"Unchanged : {unchanged}")
    print(f"Resized   : {resized}")
    print(f"Errors    : {errors}")

    # --------------------------------------------------------
    # Corrupted files
    # --------------------------------------------------------

    if corrupted_files:

        print("\n" + "-" * 70)
        print("CORRUPTED / INVALID IMAGES")
        print("-" * 70)

        for path, error in corrupted_files:

            print(f"\nFile: {path}")
            print(f"Error: {error}")

    else:

        print("\n✅ No corrupted images found.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEURAL STYLE TRANSFER DATASET PREPROCESSING")
    print("=" * 70)

    print(
        f"\nMaximum image dimension: "
        f"{MAX_SIZE}x{MAX_SIZE}"
    )

    print(
        "\nThe script will:"
        "\n1. Check images for corruption"
        "\n2. Resize very large images"
        "\n3. Preserve aspect ratio"
        "\n4. Convert resized images to JPEG"
    )

    for dataset in DATASETS:

        process_dataset(dataset)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()