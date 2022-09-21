import matplotlib.pyplot as plt
import casa_cube as casa
import cmasher as cmr
import numpy as np
import scipy.ndimage as ndimage

nv = 25
Vsyst = 6.0 #5.8  # HD163296
win = 7
PA = 154.8 -90. #43.3 # from MAPS (Huang+2018a)
plt.rcParams['font.size'] = '12'

#CO = casa.Cube('~/Observations/exoALMA/CQTau/CQ_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.image.fits')
CO = casa.Cube('~/Observations/exoALMA/DMTau/DM_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.JvMcorr.fits')
CO.plot_line()
plt.show()
#CO = casa.Cube('~/Observations/HD163296-MAPS/HD_163296_CO_220GHz.robust_0.5_wcont.image.fits')

iv0 = np.abs(CO.velocity - Vsyst).argmin()
#iv0 = 63 #np.abs(CO.velocity - Vsyst).argmin()
print("systemic channel is ",iv0," v = ",CO.velocity[iv0])

def flip_it(im,PA):
    #rotate to major axis, flip and rotate back
    im = np.fliplr(ndimage.rotate(im,PA+90.,reshape=False))
    return ndimage.rotate(im,-PA-90.-180.,reshape=False)

#def get_channel(v):
#    iv = np.abs(CO.velocity - v).argmin()

def get_vsyst(PA):
    errmin = 1e8
    for iv in range(60,65):
        channel = np.nan_to_num(CO.image[iv,:,:])
        mirror  = flip_it(channel,PA)
        fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
        axes[0].imshow(channel,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
        axes[1].imshow(mirror,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
        axes[2].imshow(channel-mirror,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
        plt.show()
        err = np.mean((channel-mirror)**2)
        if (err < errmin):
           iv00 = iv
           errmin = err
        print("error is ",err,' systemic channel is',iv00,' with v=',CO.velocity[iv00])
    return CO.velocity[iv00]

#vsyst = get_vsyst(PA)

map = np.zeros(np.shape(CO.image[iv0,:,:]))
for i in range(int(nv/2)+1):
    iv = iv0 + i - int(nv/2)

    iv_sym = iv0+(nv-1)-int(nv/2)-i
    print('channel ',iv,' goes with ',iv_sym)
    cminus = np.nan_to_num(CO.image[iv,:,:])
    cplus = np.nan_to_num(CO.image[iv_sym,:,:])

    cplusr = flip_it(cplus,PA)
    cminusr = flip_it(cminus,PA)

    fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
    axes[0].imshow(cminus,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
    axes[1].imshow(cplusr,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
    axes[2].imshow(cplusr-cminus,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
    plt.savefig('channel%i.png' % i)
    plt.pause(0.5)
    #plt.imshow(cminus+cplus,vmin=0.,vmax=0.04,origin='lower')
    #plt.show()

    if (iv==iv_sym):  # only add systemic channel once
       map = map + (cplusr-cminus)
    else:
       map = map + (cplusr-cminus) + (cminusr-cplus)

map = map/nv
print(np.min(map),np.max(map))
#plt.imshow(CO.image[0,:,:],vmin=0.,vmax=0.05,cmap='Greys')  # add continuum
plt.imshow(map,vmin=-0.002,vmax=0.002,cmap='inferno',origin='lower')
plt.savefig('HD163-moment10.png',bbox_inches='tight')
plt.show()
