import math
import os
import sys
import numpy as np
from PIL import Image, ImageFilter

# настройка
SCREEN_W, SCREEN_H = 2560, 1440
CELL_W, CELL_H = 8, 12 # размер клетки 
FONT_SIZE = 21
FPS = 30
FIT = "stretch" # "stretch" - растянуть на весь экран, "cover" - с обрезкой, "contain" - с чёрными полями***
CHARS = " .'`,:;-~+=*ilItfLCG08%#@"   
HORIZON = 0.375                    
FADE_IN = 3.5                      
DARK_CUTOFF = 0.05
SATURATION = 2.2              
BG_FILL = 0.0        


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

# хуйня, подгоняющая картинку под размер сетки, чтобы избавитсья от сжатия\растяжения
def fit_image(im, w, h):
    iw, ih = im.size
    if FIT == "stretch":
        return im.resize((w, h), Image.LANCZOS), 0.0, 1.0
    if FIT == "cover":
        s = max(w / iw, h / ih)
        nw, nh = int(iw * s), int(ih * s)
        big = im.resize((nw, nh), Image.LANCZOS)
        x0 = (nw - w) // 2
        y0 = int((nh - h) * 0.15)          # чуть выше центра, чтобы сияние попало в кадр
        return big.crop((x0, y0, x0 + w, y0 + h)), 0.0, 1.0
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    canvas = Image.new("RGB", (w, h))
    canvas.paste(im.resize((nw, nh), Image.LANCZOS), ((w - nw) // 2, (h - nh) // 2))
    return canvas, 0.0, 1.0


def load_source(path, cols, rows):
    #Загружает фото в 2x разрешении сетки + маски сияния и звёзд(доп каналы)
    im = Image.open(path).convert("RGB")
    im, _, _ = fit_image(im, cols * 2, rows * 2)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    H, W, _ = rgb.shape
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ys = (np.arange(H) / H)[:, None]
    sky = smooth((HORIZON + 0.01 - ys) / 0.05) * np.ones((1, W), np.float32)   # только небо

    #сияние
    green = g - 0.55 * np.maximum(r, b)
    pink = np.clip(r - g, 0, 1) * 1.5
    aur = np.clip(green * 3.0 + pink, 0, 1) * sky

    # звёзды: яркие точки, выступающие над размытым фоном
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    blur = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)),
                      dtype=np.float32) / 255.0
    star = np.clip((lum - blur - 0.05) * 6.0, 0, 1) * sky * (1 - aur)

    return np.dstack([rgb, aur, star]).astype(np.float32)


def bilinear(src, u, v):
    H, W, _ = src.shape
    x = np.clip(u * (W - 1), 0, W - 1.001)
    y = np.clip(v * (H - 1), 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a = src[y0, x0] * (1 - fx) + src[y0, x0 + 1] * fx
    b = src[y0 + 1, x0] * (1 - fx) + src[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


# 1 кадр
class Scene:
    def __init__(self, src, cols, rows):
        self.src, self.cols, self.rows = src, cols, rows
        self.gx, self.gy = np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows)
        rng = np.random.default_rng(7)
        self.rng = rng
        self.tw_speed = rng.uniform(1.5, 5.0, (rows, cols)).astype(np.float32)
        self.tw_phase = rng.uniform(0, 6.283, (rows, cols)).astype(np.float32)

    def frame(self, t):
        gx, gy, src = self.gx, self.gy, self.src
        two_pi = 2 * math.pi

        #течение сияния
        sky_w = smooth((HORIZON - gy) / 0.06)
        u = gx + sky_w * (0.010 * np.sin(two_pi * (1.6 * gy + 0.5 * gx) - t * 0.9)
                          + 0.006 * np.sin(two_pi * (3.1 * gy - 1.1 * gx) + t * 1.4))
        v = gy + sky_w * (0.007 * np.sin(two_pi * (2.3 * gx + 0.8 * gy) + t * 0.7)
                          + 0.004 * np.sin(two_pi * (5.0 * gx) - t * 1.9))
        s = bilinear(src, u, v)
        rgb, aur, star = s[..., :3], s[..., 3], s[..., 4]

        # яркость сияния
        gain = smooth(t / FADE_IN)
        wave = (1.0 + 0.30 * np.sin(two_pi * (2.0 * u - 0.10 * t) + 4.0 * v)
                + 0.18 * np.sin(two_pi * (5.5 * u + 0.17 * t) - 6.0 * v)
                + 0.10 * math.sin(t * 0.6))
        level = 1.0 + (wave - 1.0) * aur
        rgb = rgb * (1 - aur * (1 - gain))[..., None] * level[..., None]

        # переливы цвета
        shift = 0.5 + 0.5 * np.sin(two_pi * (1.2 * u + 0.3 * v) - t * 0.35)
        rgb[..., 2] += 0.14 * aur * shift                    # к бирюзовому
        rgb[..., 0] += 0.10 * aur * (1 - shift) * (0.5 + 0.5 * math.sin(t * 0.8))   # к розовому

        # мерцание звезд
        tw = 0.55 + 0.75 * np.sin(t * self.tw_speed + self.tw_phase)
        rgb += (star * tw)[..., None] * 0.9

        # снег ловит свет сияния 
        snow_w = smooth((gy - HORIZON) / 0.04) * np.exp(-(gy - HORIZON) * 3.0)
        glow = 0.06 * (1 + 0.5 * math.sin(t * 0.7)) * gain
        rgb[..., 1] *= 1 + glow * snow_w
        rgb[..., 0] *= 1 - 0.5 * glow * snow_w

        rgb = np.clip(rgb, 0, 1)

        # яркость -> символ (+ лёгкое мерцание букв) 
        lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        flick = (self.rng.random(lum.shape, dtype=np.float32) - 0.5) * 0.05
        idx = np.clip((lum + flick * (lum > 0.08)) * (len(CHARS) - 1) * 1.15, 0, len(CHARS) - 1).astype(np.int32)
        idx[lum < DARK_CUTOFF] = 0
        # цвет символа: ярче и насыщеннее оригинала, чтобы зелёное сияние читалось на чёрном
        bright = np.clip(rgb ** 0.6 * 1.3, 0, 1)
        grey = (bright @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        col = (np.clip(grey + (bright - grey) * SATURATION, 0, 1) * 255).astype(np.uint8)
        return idx, col


# вывод символов на экран 
def compose(idx, col, atlas):
    # векторная отрисовка сетки символов -> массив (W, H, 3) для pygame
    rows, cols = idx.shape
    ch, cw = atlas.shape[1:]
    alpha = atlas[idx].astype(np.uint16)                        # (rows, cols, ch, cw)
    img = (alpha[..., None] * col[:, :, None, None, :].astype(np.uint16)) // 255
    return img.astype(np.uint8).transpose(1, 3, 0, 2, 4).reshape(cols * cw, rows * ch, 3)


def pygame_atlas(pygame, chars, cw, ch, size):
    font = pygame.font.SysFont("consolas,couriernew,dejavusansmono,monospace", size, bold=True)
    atlas = np.zeros((len(chars), ch, cw), np.uint8)
    for i, c in enumerate(chars):
        if c == " ":
            continue
        # рисует белую букву на ЧЕРНОМ фоне без прозрачности: яркость пикселя = покрытие буквы
        surf = font.render(c, True, (255, 255, 255), (0, 0, 0)).convert()
        a = pygame.surfarray.array3d(surf)[..., 0].T               # (h, w)
        h, w = a.shape
        y0, x0 = max((ch - h) // 2, 0), max((cw - w) // 2, 0)
        hh, ww = min(h, ch), min(w, cw)
        atlas[i, y0:y0 + hh, x0:x0 + ww] = a[:hh, :ww]
    # слабая подложка под каждым символом (кроме пробела), чтобы картинка читалась плотнее
    out = (atlas.astype(np.float32) * (1 - BG_FILL) + 255 * BG_FILL).astype(np.uint8)
    out[0] = 0
    return out


def main():
    import pygame

    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "aurora.jpg")

    cols, rows = SCREEN_W // CELL_W, SCREEN_H // CELL_H
    scene = Scene(load_source(path, cols, rows), cols, rows)

    pygame.init()
    # если монитор не 2560x1440, картинка масштабируется под него
    screen = pygame.display.set_mode((cols * CELL_W, rows * CELL_H), pygame.FULLSCREEN | pygame.SCALED)
    pygame.display.set_caption("aurora")
    pygame.mouse.set_visible(False)
    atlas = pygame_atlas(pygame, CHARS, CELL_W, CELL_H, FONT_SIZE)
    clock = pygame.time.Clock()

    t, paused, running = 0.0, False, True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                t = 0.0
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
        if not paused:
            t += 1 / FPS
        idx, col = scene.frame(t)
        pygame.surfarray.blit_array(screen, compose(idx, col, atlas))
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    main()