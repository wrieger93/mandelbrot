import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numexpr as ne
import numpy as np
import PIL

# @profile
def mandelbrot(center, view_size, image_resolution, max_iters=50):
    center_re, center_im = center
    view_width, view_height = view_size
    image_horiz_res, image_vert_res = image_resolution
    total_pixels = image_horiz_res * image_vert_res

    re_axis = np.linspace(center_re - view_width/2,
                          center_re + view_width/2,
                          image_horiz_res)
    im_axis = np.linspace(center_im + view_height/2,
                          center_im - view_height/2,
                          image_vert_res)

    cs_re, cs_im = np.meshgrid(re_axis, im_axis)
    cs_re = cs_re.flatten()
    cs_im = cs_im.flatten()

    zs_re = np.zeros(total_pixels)
    zs_im = np.zeros(total_pixels)

    image = np.zeros(total_pixels, dtype=np.int)
    image_coords = np.arange(0, total_pixels, dtype=np.int)

    divergent = np.zeros(total_pixels, dtype=np.bool)

    for i in range(1, 51):
        zs_re, zs_im = (zs_re * zs_re) - (zs_im * zs_im) + cs_re, 2*(zs_re * zs_im) + cs_im
        divergent = (zs_re * zs_re) + (zs_im * zs_im) > 4
        image[image_coords[divergent]] = i

        not_divergent = np.logical_not(divergent)
        zs_re = zs_re[not_divergent]
        zs_im = zs_im[not_divergent]
        cs_re = cs_re[not_divergent]
        cs_im = cs_im[not_divergent]
        image_coords = image_coords[not_divergent]

    return image.reshape(image_resolution)


def mandelbrot_numexpr(center, view_size, image_resolution, max_iters=50):
    center_re, center_im = center
    view_width, view_height = view_size
    image_horiz_res, image_vert_res = image_resolution
    total_pixels = image_horiz_res * image_vert_res

    re_axis = np.linspace(center_re - view_width/2,
                          center_re + view_width/2,
                          image_horiz_res)
    im_axis = np.linspace(center_im + view_height/2,
                          center_im - view_height/2,
                          image_vert_res)

    cs_re, cs_im = np.meshgrid(re_axis, im_axis)
    cs_re = np.double(cs_re.flatten())
    cs_im = np.double(cs_im.flatten())

    zs_re = np.zeros(total_pixels, dtype=np.double)
    zs_im = np.zeros(total_pixels, dtype=np.double)

    image = np.zeros(total_pixels, dtype=np.int)
    image_coords = np.arange(0, total_pixels, dtype=np.int)

    for i in range(1, 51):
        zs_re, zs_im = ne.evaluate("(zs_re * zs_re) - (zs_im * zs_im) + cs_re"), ne.evaluate("2*(zs_re * zs_im) + cs_im")
        divergent = ne.evaluate("(zs_re * zs_re) + (zs_im * zs_im) > 4")
        image[image_coords[divergent]] = i

        # -divergent is equivalent to np.logical_not(divergent)
        # for bool arrays
        not_divergent = np.logical_not(divergent)
        zs_re = zs_re[not_divergent]
        zs_im = zs_im[not_divergent]
        cs_re = cs_re[not_divergent]
        cs_im = cs_im[not_divergent]
        image_coords = image_coords[not_divergent]

    return image.reshape(image_resolution)


def mandelbrot_numexpr_complex(center, view_size, image_resolution, max_iters=50):
    center_re, center_im = center
    view_width, view_height = view_size
    image_horiz_res, image_vert_res = image_resolution
    total_pixels = image_horiz_res * image_vert_res

    re_axis = np.linspace(center_re - view_width/2,
                          center_re + view_width/2,
                          image_horiz_res,
                          dtype=np.float_)
    im_axis = np.linspace(center_im + view_height/2,
                          center_im - view_height/2,
                          image_vert_res,
                          dtype=np.float_)

    cs_re, cs_im = np.meshgrid(re_axis, im_axis)
    cs = (cs_re + 1j*cs_im).flatten()

    zs = np.zeros(total_pixels, dtype=np.complex_)

    image = np.zeros(total_pixels, dtype=np.int_)
    image_coords = np.arange(0, total_pixels, dtype=np.int_)

    for i in range(1, max_iters+1):
        # zs = ne.evaluate("zs*zs + cs")
        # divergent = ne.evaluate("real(abs(zs)) > 2")
        zs = zs*zs + cs
        divergent = np.absolute(zs) > 2
        image[image_coords[divergent]] = i

        not_divergent = np.logical_not(divergent)
        zs = zs[not_divergent]
        cs = cs[not_divergent]
        image_coords = image_coords[not_divergent]

    return image.reshape(image_resolution)


def mandelbrot_one_row(cs, max_iters=50):
    total_pixels = cs.size

    zs = np.zeros(total_pixels, dtype=np.complex_)

    image_row = np.zeros(total_pixels, dtype=np.int_)
    image_coords = np.arange(0, total_pixels, dtype=np.int_)

    for i in range(1, max_iters+1):
        zs = zs*zs + cs
        divergent = np.absolute(zs) > 2
        image_row[image_coords[divergent]] = i

        not_divergent = np.logical_not(divergent)
        zs = zs[not_divergent]
        cs = cs[not_divergent]
        image_coords = image_coords[not_divergent]

    return image_row

def mandelbrot_mp(center, view_size, image_resolution, max_iters=50):
    center_re, center_im = center
    view_width, view_height = view_size
    image_horiz_res, image_vert_res = image_resolution
    total_pixels = image_horiz_res * image_vert_res

    re_axis = np.linspace(center_re - view_width/2,
                          center_re + view_width/2,
                          image_horiz_res,
                          dtype=np.float_)
    im_axis = np.linspace(center_im + view_height/2,
                          center_im - view_height/2,
                          image_vert_res,
                          dtype=np.float_)

    cs_re, cs_im = np.meshgrid(re_axis, im_axis)
    cs = cs_re + 1j*cs_im

    pool = mp.Pool(processes=4)
    results = pool.map(mandelbrot_one_row, cs)
    return np.array(results)

def f(a, b):
    print("{a} + {b} = {c}".format(a=a, b=b, c=a+b))

if __name__ == "__main__":
    center = (0.274, 0.482)
    view_size = (0.05, 0.05)
    image_resolution = (1000, 1000)
    image_data = mandelbrot_mp(center, view_size, image_resolution, max_iters=50)
    colormap = cm.ScalarMappable(cmap="Blues")
    image = PIL.Image.fromarray(colormap.to_rgba(image_data, bytes=True)[:,:,:3])
    image.save("quad_spiral_valley_small.png")
