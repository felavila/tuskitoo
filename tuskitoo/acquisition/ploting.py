import numpy as np 
import matplotlib.pyplot as plt 


def arrow_plot(ax,data,angulo_radianes=0):
   """angulo_radianes angulo de rotacion de la imagen respecto al cielo se puede sacar del header
   by this rotation_angle = np.arctan(CD2_1 / CD1_1) -> rotation angle respect to the sky"""
   arrow_length = min(data.shape) * 0.1  # Arrow length covers 10% of the smaller dimension
   # Calculate the starting position of the arrows
   start_x = data.shape[0] * 0.2
   start_y = data.shape[0] * 0.2
   # Calculate the end positions of the arrows
   angulo_radianes = angulo_radianes # 1.5708 90 ° in grade

   # Add the first arrow
   arrow_head_width = arrow_head_length = arrow_length * 0.2  # Arrowhead size is 20% of arrow length
   
   
   ax.arrow(start_x, start_y, arrow_length* np.cos(angulo_radianes+1.5708), arrow_length* np.sin(angulo_radianes+1.5708), color='red', head_width=arrow_head_width, head_length=arrow_head_length)
   # Add the second arrow perpendicular to the first arrow
   ax.arrow(start_x, start_y,  arrow_length* np.cos(angulo_radianes+1.5708*2),  arrow_length* np.sin(angulo_radianes+1.5708*2), color='red', head_width=arrow_head_width, head_length=arrow_head_length)

   # Add text at the end points of each arrow
   text_offset = 0.1 # Offset for positioning text
   end_x_blue = (start_x) + 1.4*arrow_length* np.cos(angulo_radianes+1.5708)
   end_y_blue = (start_y) + 1.4*arrow_length* np.sin(angulo_radianes+1.5708)
   ax.text(end_x_blue, end_y_blue, "N", ha="center", va='center' ,color='red')
   end_x_red = (start_x) + 1.4*arrow_length* np.cos(angulo_radianes+1.5708*2)
   end_y_red = (start_y) + 1.4*arrow_length* np.sin(angulo_radianes+1.5708*2)
   ax.text(end_x_red,end_y_red, 'E', ha='center', va='center', color='red')



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