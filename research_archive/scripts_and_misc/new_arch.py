class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        mid = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False), nn.ReLU(),
            nn.Conv2d(mid, in_planes, 1, bias=False))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=(ks-1)//2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.ca = ChannelAttention(p)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BiFPNLayer(nn.Module):
    def __init__(self, ch=256, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.w_p4_td  = nn.Parameter(torch.ones(2))
        self.w_p3_out = nn.Parameter(torch.ones(2))
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(2))
        self.conv_p4_td  = DWConv(ch)
        self.conv_p3_out = DWConv(ch)
        self.conv_p4_out = DWConv(ch)
        self.conv_p5_out = DWConv(ch)

    def _up(self, x, t):
        return F.interpolate(x, t.shape[-2:], mode='nearest')

    def _dn(self, x, t):
        return F.adaptive_avg_pool2d(x, t.shape[-2:])

    def forward(self, p3, p4, p5):
        w4  = F.relu(self.w_p4_td.clone());  w4  = w4  / (w4.sum()  + self.eps)
        w3  = F.relu(self.w_p3_out.clone()); w3  = w3  / (w3.sum()  + self.eps)
        w4o = F.relu(self.w_p4_out.clone()); w4o = w4o / (w4o.sum() + self.eps)
        w5o = F.relu(self.w_p5_out.clone()); w5o = w5o / (w5o.sum() + self.eps)
        p4_td  = self.conv_p4_td(w4[0]*p4  + w4[1]*self._up(p5, p4))
        p3_out = self.conv_p3_out(w3[0]*p3 + w3[1]*self._up(p4_td, p3))
        p4_out = self.conv_p4_out(w4o[0]*p4 + w4o[1]*p4_td + w4o[2]*self._dn(p3_out, p4))
        p5_out = self.conv_p5_out(w5o[0]*p5 + w5o[1]*self._dn(p4_out, p5))
        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    def __init__(self, in_ch, out_ch=256, n=2):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.SiLU())
            for c in in_ch
        ])
        self.layers = nn.ModuleList([BiFPNLayer(out_ch) for _ in range(n)])

    def forward(self, p3r, p4r, p5r):
        p3, p4, p5 = self.lat[0](p3r), self.lat[1](p4r), self.lat[2](p5r)
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(self.eps).pow(self.p), x.shape[-2:]
        ).pow(1.0 / self.p)


class DRTeacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg['backbone'], pretrained=False,
            features_only=True, out_indices=(2, 3, 4),
            drop_path_rate=cfg['drop_path_rate'])

        ch = self.backbone.feature_info.channels()
        oc = cfg['bifpn_channels']
        self.bifpn    = BiFPN(ch, oc, cfg['bifpn_layers'])
        self.pool     = GeMPooling()
        self.cbam_p3  = CBAM(oc)
        self.cbam_p5  = CBAM(oc)
        self.dropout  = nn.Dropout(cfg['dropout'])
        self.head     = nn.Linear(oc * 2, CORAL_LEVELS)
        self.msd_k    = cfg['msd_k']

    def forward(self, x):
        f = self.backbone(x)
        p3, p4, p5 = self.bifpn(f[0], f[1], f[2])
        feat = torch.cat([
            self.pool(self.cbam_p3(p3)).flatten(1),
            self.pool(self.cbam_p5(p5)).flatten(1)
        ], 1)
        if self.training:
            return torch.stack([
                self.head(self.dropout(feat)) for _ in range(self.msd_k)
            ]).mean(0)
        return self.head(feat)
