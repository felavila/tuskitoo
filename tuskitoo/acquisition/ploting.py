import numpy as np 
import matplotlib.pyplot as plt 


def arrow_plot(ax, angulo_radianes=0, frac_pos=0.15, frac_len=0.11, color="k"):
    """
    Plot N/E arrows using the current axis limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the arrows are drawn.
    angulo_radianes : float, optional
        Rotation angle of the image with respect to the sky, in radians.
    frac_pos : float, optional
        Fractional position inside the axes for the arrow origin.
    frac_len : float, optional
        Fraction of the smallest axis span used as arrow length.
    color : str, optional
        Arrow/text color.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmin, xmax = min(xlim), max(xlim)
    ymin, ymax = min(ylim), max(ylim)

    dx = xmax - xmin
    dy = ymax - ymin

    # Start position near lower-left corner of current view
    start_x = xmin + frac_pos * dx
    start_y = ymin + frac_pos * dy

    # Arrow length scaled to current displayed region
    arrow_length = frac_len * min(dx, dy)
    head_size = 0.2 * arrow_length

    # North direction
    theta_n = angulo_radianes + np.pi / 2
    dx_n = arrow_length * np.cos(theta_n)
    dy_n = arrow_length * np.sin(theta_n)

    # East direction
    theta_e = angulo_radianes + np.pi
    dx_e = arrow_length * np.cos(theta_e)
    dy_e = arrow_length * np.sin(theta_e)

    ax.arrow(
        start_x, start_y, dx_n, dy_n,
        color=color, head_width=head_size, head_length=head_size,
        length_includes_head=True
    )
    ax.arrow(
        start_x, start_y, dx_e, dy_e,
        color=color, head_width=head_size, head_length=head_size,
        length_includes_head=True,zorder=10
    )

    # Labels
    text_scale = 1.35
    ax.text(
        start_x + text_scale * dx_n,
        start_y + text_scale * dy_n,
        "N",
        color=color,
        ha="center",
        va="center",
    )
    ax.text(
        start_x + text_scale * dx_e,
        start_y + text_scale * dy_e,
        "E",
        color=color,
        ha="center",
        va="center",
    )



def plot_image_cut(full_image,cut_image,sky_coords=None,gaia_coords=None,object_cords=None,title=None):
   fig = plt.figure(figsize=(25, 10))
   axis1 = fig.add_subplot(1, 2, 1, projection=full_image.wcs)
   axis2 = fig.add_subplot(1, 2, 2, projection=cut_image.wcs)
   axis1.imshow(np.log10(full_image.data),origin="lower", cmap=plt.cm.viridis)#,vmin=np.quantile(np.log10(shifted_data),0.45),vmax=np.quantile(np.log10(shifted_data),1))
   axis2.imshow(np.log10(cut_image.data),origin="lower", cmap=plt.cm.viridis)
   if sky_coords:
      pixel_full = np.column_stack(full_image.wcs.world_to_pixel(sky_coords))
      pixel_cut = np.column_stack(cut_image.wcs.world_to_pixel(sky_coords))
      for i in pixel_full:
         axis1.scatter(*i,color="r")
      for i in pixel_cut:
         axis2.scatter(*i,color="r")
   if gaia_coords:
      gaia_full = np.column_stack(full_image.wcs.world_to_pixel(gaia_coords))
      gaia_cut = np.column_stack(cut_image.wcs.world_to_pixel(gaia_coords))
      for i in gaia_full:
         if full_image.data.shape[0]<i[0] or full_image.data.shape[1]<i[1] or i[0]<0 or i[1]<0:
            continue
         axis1.scatter(*i,color="k")
      for i in gaia_cut:
         if cut_image.data.shape[0]<i[0] or cut_image.data.shape[1]<i[1] or i[0]<0 or i[1]<0:
            continue
         axis2.scatter(*i,color="k")
   if object_cords:
      print("TODO")
   plt.suptitle(title, fontsize=25)
   axis1.coords['ra'].set_axislabel('Right Ascension')
   axis1.coords['dec'].set_axislabel('Declination')
   axis2.coords['ra'].set_axislabel('Right Ascension')
   axis2.coords['dec'].set_axislabel('Declination')
   axis1.set_xlabel(axis1.get_xlabel(), fontsize=20)
   axis1.set_ylabel(axis1.get_ylabel(),     fontsize=20)
   axis2.set_xlabel(axis2.get_xlabel(), fontsize=20)
   axis2.set_ylabel(axis2.get_xlabel(),     fontsize=20)
   axis1.tick_params(axis='both', which='major', labelsize=20)
   axis2.tick_params(axis='both', which='major', labelsize=20)
   plt.show()
        # axis1.imshow(np.log10(image),origin="lower", cmap=plt.cm.viridis)#,vmin=np.quantile(np.log10(shifted_data),0.45),vmax=np.quantile(np.log10(shifted_data),1))
        # axis2.imshow(np.log10(cutout_2d.data),origin="lower", cmap=plt.cm.viridis)
        # #sky_big = np.column_stack(wcs.world_to_pixel(coords_sky))
        # #for i in sky_big:
        # #   axis1.scatter(*i,color="r")
        # if gaia_coords:
        #     gaia_pixel_positions = np.column_stack(wcs.world_to_pixel(gaia_coords))
        #     for i in gaia_pixel_positions:
        #         if image.shape[0]<i[0] or image.shape[1]<i[1] or i[0]<0 or i[1]<0:
        #             continue
        #         axis1.scatter(*i,color="k")   
        # for i in coords_pixel.T:
        #     axis2.scatter(*i,color="r")
        # if gaia_coords:
        #     gaia_pixel_positions = np.column_stack(cutout_2d.wcs.world_to_pixel(gaia_coords[idx][good_matches]))
        #     for i in gaia_pixel_positions:
        #         axis2.scatter(*i,color="k")
        # plt.show()