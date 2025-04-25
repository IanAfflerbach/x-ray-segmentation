from PIL import Image, ImageChops

def save_model(model, name):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))

def load_model(model_class, name):
    from torch import load
    from os import path
    r = model_class()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r

def segment_image(img, mask):
    img = img.convert('RGB')
    mask = mask.convert('RGB')
    return ImageChops.multiply(img, mask)

def stitch_images_horizontal(*imgs):
    assert len(imgs) >= 2, "Must pass 2 or more PIL images"

    imgl = imgs[0]
    imgr = None

    if len(imgs) > 2:
        imgr = stitch_images_horizontal(*imgs[1:])
    else:
        imgr = imgs[1]
    
    result_width = imgl.width + imgr.width
    stitch = Image.new('RGB', (result_width, imgl.height))
    stitch.paste(im=imgl, box=(0, 0))
    stitch.paste(im=imgr, box=(imgl.width, 0))
    return stitch

def stitch_images_vertical(*imgs):
    assert len(imgs) >= 2, "Must pass 2 or more PIL images"

    imgl = imgs[0]
    imgr = None

    if len(imgs) > 2:
        imgr = stitch_images_vertical(*imgs[1:])
    else:
        imgr = imgs[1]
    
    result_height = imgl.height + imgr.height
    stitch = Image.new('RGB', (imgl.width, result_height))
    stitch.paste(im=imgl, box=(0, 0))
    stitch.paste(im=imgr, box=(0, imgl.height))
    return stitch