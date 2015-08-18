import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import PIL

# @profile
def mandelbrot(center, view_size, image_resolution):
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


def mandelbrot_numexpr(center, view_size, image_resolution):
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


def mandelbrot_numexpr_complex(center, view_size, image_resolution):
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

    for i in range(1, 51):
        zs = ne.evaluate("zs*zs + cs")
        divergent = ne.evaluate("real(abs(zs)) > 2")
        image[image_coords[divergent]] = i

        not_divergent = np.logical_not(divergent)
        zs = zs[not_divergent]
        cs = cs[not_divergent]
        image_coords = image_coords[not_divergent]

    return image.reshape(image_resolution)


if __name__ == "__main__":
    center = (-0.4, 0.1)
    view_size = (1, 1)
    image_resolution = (1000, 1000)
    # image = mandelbrot((0,0), (4,4), (10000,10000))
