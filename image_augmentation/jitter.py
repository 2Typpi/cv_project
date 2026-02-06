from PIL import Image, ImageDraw, ImageEnhance
import random


# Range depended functions 
def apply_geometric_jitter(image, rotation_limit=10.0, translation_limit=5, perspective_limit=0.02):
    """
    Applies geometric jitter to an image, including rotation, translation, and perspective shifts using random values within limits.
    """
    width, height = image.size
    
    angle = random.uniform(-rotation_limit, rotation_limit)
    tx = random.uniform(-translation_limit, translation_limit)
    ty = random.uniform(-translation_limit, translation_limit)

    img = image.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty))

    coeffs = [
        1 + random.uniform(-perspective_limit, perspective_limit), 0, 0,
        0, 1 + random.uniform(-perspective_limit, perspective_limit), 0,
        random.uniform(-0.0001, 0.0001), random.uniform(-0.0001, 0.0001)
    ]
    
    return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

def apply_brightness_jitter_range(image, jitter_range=(0.7, 1.3)):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(*jitter_range)
    return enhancer.enhance(factor)

def jitter_image_random(img, rot_limit=5, trans_limit=3, persp_limit=0.02, bright_factor=1.0, bright_range_delta=0.3):
    """
    Applies a combination of geometric and brightness jitter to an image.
    """
    img = apply_geometric_jitter(img, rotation_limit=rot_limit, translation_limit=trans_limit, perspective_limit=persp_limit)
    
    brightness_min = max(0.1, bright_factor - bright_range_delta)
    brightness_max = min(2.0, bright_factor + bright_range_delta)
    
    img = apply_brightness_jitter_range(img, jitter_range=(brightness_min, brightness_max))
    return img

# Specific augmentations
def apply_geometric_transform(image, angle, tx, ty, perspective_coeffs):
    width, height = image.size
    img = image.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty))
    coeffs = [
        1 + perspective_coeffs, 0, 0,
        0, 1 + perspective_coeffs, 0,
        0, 0
    ]
    return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

def apply_brightness_jitter(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def jitter_image(img, angle, tx, ty, perspective_coeffs, brightness_factor):
    """
    Applies a specific combination of geometric and brightness jitter to an image.
    """
    img = apply_geometric_transform(img, angle, tx, ty, perspective_coeffs)
    img = apply_brightness_jitter(img, brightness_factor)
    return img

# Split functions
def split_image_diagonal_random(image_path, min_overlap_pct=0.1):
    """
    Splits an image diagonally into two parts, including a certain overlap.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    margin = 50 
    
    top_x = random.randint(margin, w - margin)
    
    max_slant = w
    min_overlap = int(w * min_overlap_pct)
    bottom_x = random.randint(max(margin, top_x - max_slant), 
                              min(w - margin, top_x + max_slant))
    
    mask_left = Image.new("L", (w, h), 0)
    draw_l = ImageDraw.Draw(mask_left)
    draw_l.polygon([(0, 0), (top_x + min_overlap, 0), (bottom_x + min_overlap, h), (0, h)], fill=255)
    
    mask_right = Image.new("L", (w, h), 0)
    draw_r = ImageDraw.Draw(mask_right)
    draw_r.polygon([(top_x - min_overlap, 0), (w, 0), (w, h), (bottom_x - min_overlap, h)], fill=255)

    left_img = img.copy()
    left_img.putalpha(mask_left)
    
    right_img = img.copy()
    right_img.putalpha(mask_right)

    cropped_left = remove_alpha(left_img)
    cropped_right = remove_alpha(right_img)

    return cropped_left, cropped_right, img

def split_image_diagonal(image_path, min_overlap_pct):
    """
    Splits an image diagonally into two parts.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    margin = 50

    top_x = w // 2
    bottom_x = w // 2

    min_overlap = int(w * min_overlap_pct)

    mask_left = Image.new("L", (w, h), 0)
    draw_l = ImageDraw.Draw(mask_left)
    draw_l.polygon([(0, 0), (top_x + min_overlap, 0), (bottom_x + min_overlap, h), (0, h)], fill=255)

    mask_right = Image.new("L", (w, h), 0)
    draw_r = ImageDraw.Draw(mask_right)
    draw_r.polygon([(top_x - min_overlap, 0), (w, 0), (w, h), (bottom_x - min_overlap, h)], fill=255)

    left_img = img.copy()
    left_img.putalpha(mask_left)

    right_img = img.copy()
    right_img.putalpha(mask_right)

    cropped_left = remove_alpha(left_img)
    cropped_right = remove_alpha(right_img)

    return cropped_left, cropped_right, img


def remove_alpha(img_rgba, bg_color=(0, 0, 0)):
    """
    Removes the alpha channel from an RGBA image and replaces it with a solid background color.
    """
    background = Image.new("RGB", img_rgba.size, bg_color)
    background.paste(img_rgba, mask=img_rgba.split()[3]) 
    return background