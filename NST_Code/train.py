
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from utils.utils import *
from utils.models import *

import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    # --------------------------------------------------
    # Dataset and model paths
    # --------------------------------------------------

    parser.add_argument(
        '--content_dir',
        type=str,
        default=r'D:\Projects\Neural Style Transfer\NST_Code\content_data',
        help='Location of content dataset'
    )

    parser.add_argument(
        '--style_dir',
        type=str,
        default=r'D:\Projects\Neural Style Transfer\NST_Code\style_data',
        help='Location of style dataset'
    )

    parser.add_argument(
        '--vgg',
        type=str,
        default=r'D:\Projects\Neural Style Transfer\NST_Code\vgg_normalised.pth',
        help='Location of pre-trained VGG'
    )

    # --------------------------------------------------
    # Experiment
    # --------------------------------------------------

    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment1',
        help='Name of experiment'
    )

    # --------------------------------------------------
    # Image sizes
    # --------------------------------------------------

    parser.add_argument(
        '--final_size',
        type=int,
        default=512,
        help='Size of final image'
    )

    parser.add_argument(
        '--content_size',
        type=int,
        default=265,
        help='Size of content image'
    )

    parser.add_argument(
        '--style_size',
        type=int,
        default=256,
        help='Size of style image'
    )

    # --------------------------------------------------
    # Image processing
    # --------------------------------------------------

    parser.add_argument(
        '--crop',
        action='store_true',
        default=True,
        help='Crop image'
    )

    # --------------------------------------------------
    # Training parameters
    # --------------------------------------------------

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--lr_decay',
        type=float,
        default=5e-5,
        help='Learning rate decay'
    )

    parser.add_argument(
        '--epoch',
        type=int,
        default=2,
        help='Number of epochs'
    )

    parser.add_argument(
        '--content_weight',
        type=float,
        default=1.0,
        help='Content loss weight'
    )

    parser.add_argument(
        '--style_weight',
        type=float,
        default=10.0,
        help='Style loss weight'
    )

    # --------------------------------------------------
    # Logging and saving
    # --------------------------------------------------

    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='Log interval'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=2,
        help='Save interval'
    )

    # --------------------------------------------------
    # Resume training
    # --------------------------------------------------

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume training'
    )

    parser.add_argument(
        '--decoder_path',
        type=str,
        default=None,
        help='Path to decoder checkpoint'
    )

    parser.add_argument(
        '--optimizer_path',
        type=str,
        default=None,
        help='Path to optimizer checkpoint'
    )

    return parser.parse_args()


def main():

    # ==================================================
    # 1. Parse arguments
    # ==================================================

    args = parse_arguments()

    # ==================================================
    # 2. Select device
    # ==================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ==================================================
    # 3. Create experiment directory
    # ==================================================

    saved_dir = Path('experiment') / args.experiment

    saved_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    # Save arguments
    with open(saved_dir / 'args.txt', 'w') as args_file:

        for key, value in vars(args).items():
            args_file.write(
                f'{key}: {value}\n'
            )

    # ==================================================
    # 4. Create transforms
    # ==================================================

    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.final_size
    )

    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.final_size
    )

    # ==================================================
    # 5. Create datasets
    # ==================================================

    content_dataset = ImageFolderDataset(
        args.content_dir,
        content_transform
    )

    style_dataset = ImageFolderDataset(
        args.style_dir,
        style_transform
    )

    print(
        f"Number of content images: {len(content_dataset)}"
    )

    print(
        f"Number of style images: {len(style_dataset)}"
    )

    # ==================================================
    # 6. Create DataLoaders
    # ==================================================

    pin_memory = torch.cuda.is_available()

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True
    )

    print(
        "Number of batches in content dataset:",
        len(content_dataloader)
    )

    print(
        "Number of batches in style dataset:",
        len(style_dataloader)
    )

    # ==================================================
    # 7. Create VGG Encoder
    # ==================================================

    encoder = VGGEncoder(
        args.vgg
    ).to(device)

    # ==================================================
    # 8. Create Decoder
    # ==================================================

    decoder = Decoder().to(device)

    # ==================================================
    # 9. Create optimizer
    # ==================================================

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    # ==================================================
    # 10. Learning rate scheduler
    # ==================================================

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch:
        1.0 / (1.0 + args.lr_decay * epoch)
    )

    # ==================================================
    # 11. Resume training
    # ==================================================

    if args.resume:

        if args.decoder_path is None:
            raise ValueError(
                "Please provide --decoder_path when using --resume"
            )

        if args.optimizer_path is None:
            raise ValueError(
                "Please provide --optimizer_path when using --resume"
            )

        decoder_path = Path(args.decoder_path)
        optimizer_path = Path(args.optimizer_path)

        if not decoder_path.exists():
            raise FileNotFoundError(
                f"Decoder checkpoint not found: {decoder_path}"
            )

        if not optimizer_path.exists():
            raise FileNotFoundError(
                f"Optimizer checkpoint not found: {optimizer_path}"
            )

        print("Loading decoder checkpoint...")

        decoder.load_state_dict(
            torch.load(
                decoder_path,
                map_location=device
            )
        )

        print("Loading optimizer checkpoint...")

        optimizer.load_state_dict(
            torch.load(
                optimizer_path,
                map_location=device
            )
        )

        print("Training resumed successfully.")

    # ==================================================
    # 12. Loss function
    # ==================================================

    mse_loss = torch.nn.MSELoss()

    # ==================================================
    # 13. Freeze VGG
    # ==================================================

    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    # ==================================================
    # 14. Number of batches
    # ==================================================

    num_batches = min(
        len(content_dataloader),
        len(style_dataloader)
    )

    if num_batches == 0:
        raise ValueError(
            "Dataset is too small for the selected batch size. "
            "Try reducing --batch_size."
        )

    # ==================================================
    # 15. Training loop
    # ==================================================

    for epoch in range(args.epoch):

        decoder.train()

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        progress_bar = tqdm(
            zip(
                content_dataloader,
                style_dataloader
            ),
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{args.epoch}"
        )

        # --------------------------------------------------
        # Batch loop
        # --------------------------------------------------

        for content_batch, style_batch in progress_bar:

            # ==============================================
            # Move images to device
            # ==============================================

            content_batch = content_batch.to(
                device,
                non_blocking=True
            )

            style_batch = style_batch.to(
                device,
                non_blocking=True
            )

            # ==============================================
            # Extract Content and Style Features
            # ==============================================

            with torch.no_grad():

                c_feats = encoder(
                    content_batch
                )

                s_feats = encoder(
                    style_batch
                )

            # ==============================================
            # Adaptive Instance Normalization
            # ==============================================

            t = adaptive_instance_normalization(
                c_feats[-1],
                s_feats[-1]
            )

            # ==============================================
            # Decode AdaIN feature
            # ==============================================

            g = decoder(t)

            # ==============================================
            # Extract generated image features
            #
            # IMPORTANT:
            # We do NOT use torch.no_grad() here because
            # gradients need to flow back to the decoder.
            # ==============================================

            g_features = encoder(g)

            # ==============================================
            # Content Loss
            # ==============================================

            loss_c = (
                mse_loss(
                    g_features[-1],
                    t
                )
                * args.content_weight
            )

            # ==============================================
            # Style Loss
            # ==============================================

            loss_s = 0.0

            for g_f, s_f in zip(
                g_features,
                s_feats
            ):

                # Generated image statistics
                g_mean, g_std = calc_mean_std(
                    g_f
                )

                # Style image statistics
                s_mean, s_std = calc_mean_std(
                    s_f
                )

                # Mean loss
                mean_loss = mse_loss(
                    g_mean,
                    s_mean
                )

                # Standard deviation loss
                std_loss = mse_loss(
                    g_std,
                    s_std
                )

                loss_s += (
                    mean_loss + std_loss
                )

            loss_s *= args.style_weight

            # ==============================================
            # Total Loss
            # ==============================================

            loss = loss_c + loss_s

            # ==============================================
            # Backpropagation
            # ==============================================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # ==============================================
            # Running Loss
            # ==============================================

            running_loss += loss.item()

            running_closs += loss_c.item()

            running_sloss += loss_s.item()

            # ==============================================
            # Progress Bar
            # ==============================================

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                content=f"{loss_c.item():.4f}",
                style=f"{loss_s.item():.4f}"
            )

        # ==================================================
        # 16. Update learning rate
        # ==================================================

        scheduler.step()

        # ==================================================
        # 17. Calculate average losses
        # ==================================================

        avg_loss = (
            running_loss / num_batches
        )

        avg_closs = (
            running_closs / num_batches
        )

        avg_sloss = (
            running_sloss / num_batches
        )

        current_lr = optimizer.param_groups[0]['lr']

        # ==================================================
        # 18. Print epoch information
        # ==================================================

        if (epoch + 1) % args.log_interval == 0:

            tqdm.write(
                f"\n"
                f"Epoch {epoch + 1}/{args.epoch}\n"
                f"Loss          : {avg_loss:.4f}\n"
                f"Content Loss  : {avg_closs:.4f}\n"
                f"Style Loss    : {avg_sloss:.4f}\n"
                f"Learning Rate : {current_lr:.8f}\n"
            )

        # ==================================================
        # 19. Save checkpoint
        # ==================================================

        if (epoch + 1) % args.save_interval == 0:

            decoder_checkpoint = (
                saved_dir /
                f"decoder_{epoch + 1}.pth"
            )

            optimizer_checkpoint = (
                saved_dir /
                f"optimizer_{epoch + 1}.pth"
            )

            torch.save(
                decoder.state_dict(),
                decoder_checkpoint
            )

            torch.save(
                optimizer.state_dict(),
                optimizer_checkpoint
            )

            print(
                f"Decoder saved to: "
                f"{decoder_checkpoint}"
            )

            print(
                f"Optimizer saved to: "
                f"{optimizer_checkpoint}"
            )

            # ==================================================
            # 20. Save sample output
            # ==================================================

            decoder.eval()

            with torch.no_grad():

                output = torch.cat(
                    [
                        content_batch,
                        style_batch,
                        g
                    ],
                    dim=0
                )

                output_path = (
                    saved_dir /
                    f"output_{epoch + 1}.jpg"
                )

                save_image(
                    output,
                    output_path,
                    nrow=args.batch_size
                )

            print(
                f"Sample output saved to: "
                f"{output_path}"
            )

    # ==================================================
    # Training finished
    # ==================================================

    print("\nTraining completed!")




if __name__ == '__main__':
    main()

