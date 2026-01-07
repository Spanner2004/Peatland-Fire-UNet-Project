import torch
import torch.nn as nn
# Base U-Net implementation adapted from: https://github.com/mtran5/UNet

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding="same", dropout=0.0):
        super().__init__()
        pad = 1 if padding == "same" else padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if (dropout and dropout > 0.0) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans, layers=2, sampling_factor=2, padding="same", dropout=0.0):
        super().__init__()
        blocks = []
        blocks.append(ConvBlock(in_chans, out_chans, padding=padding, dropout=dropout))
        for _ in range(layers - 1):
            blocks.append(ConvBlock(out_chans, out_chans, padding=padding, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(kernel_size=sampling_factor)

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.pool(feat)
        return pooled, feat

class DecoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans, layers=2, sampling_factor=2, padding="same", dropout=0.0, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.tconv = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=sampling_factor, stride=sampling_factor)
        conv_in = (in_chans // 2) + (out_chans if use_skip else 0)
        blocks = []
        blocks.append(ConvBlock(conv_in, out_chans, padding=padding, dropout=dropout))
        for _ in range(layers - 1):
            blocks.append(ConvBlock(out_chans, out_chans, padding=padding, dropout=dropout))
        self.decoder = nn.Sequential(*blocks)

    def forward(self, x, skip_feat=None):
        x = self.tconv(x)
        if self.use_skip and skip_feat is not None:
            if skip_feat.size(-2) != x.size(-2) or skip_feat.size(-1) != x.size(-1):
                _, _, sh, sw = skip_feat.shape
                _, _, h, w = x.shape
                top = (sh - h) // 2
                left = (sw - w) // 2
                skip_feat = skip_feat[:, :, top:top+h, left:left+w]
            x = torch.cat([skip_feat, x], dim=1)
        x = self.decoder(x)
        return x

class SingleHeadUNet(nn.Module):
    def __init__(self,
                 in_chans=7,
                 base_filters=32,
                 depth=5,
                 layers=2,
                 sampling_factor=2,
                 padding="same",
                 dropout=0.05,
                 use_skip=True):
        """
        Single output UNet that returns one raw logit channel for mask.

        Parameters
        in_chans int number of input channels
        base_filters int filters at first encoder stage
        depth int number of encoder stages
        layers int convs per block
        dropout float dropout probability inside conv blocks
        use_skip bool toggle skip connections
        """
        super().__init__()

        self.use_skip = use_skip

        # encoder
        self.encoder_blocks = nn.ModuleList()
        in_c = in_chans
        filters = base_filters
        for _ in range(depth):
            self.encoder_blocks.append(EncoderBlock(in_c, filters, layers=layers, sampling_factor=sampling_factor, padding=padding, dropout=dropout))
            in_c = filters
            filters = filters * 2

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(in_c, in_c * 2, padding=padding, dropout=dropout),
            ConvBlock(in_c * 2, in_c * 2, padding=padding, dropout=dropout)
        )
        bottleneck_channels = in_c * 2

        # decoder
        self.decoder_blocks = nn.ModuleList()
        in_ch = bottleneck_channels
        out_ch = in_ch // 2
        for _ in range(depth - 1):
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch, layers=layers, sampling_factor=sampling_factor, padding=padding, dropout=dropout, use_skip=use_skip))
            in_ch = out_ch
            out_ch = max(out_ch // 2, base_filters)

        # final conv to 1 logit channel
        self.logits = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encoder_blocks:
            out, skip = enc(out)
            skips.append(skip)
        out = self.bottleneck(out)

        # decode using index based skip access so shapes remain intact
        for i, db in enumerate(self.decoder_blocks):
            skip_idx = -(i + 1)
            skip_feat = skips[skip_idx] if self.use_skip else None
            out = db(out, skip_feat)

        logits = self.logits(out)
        return logits
