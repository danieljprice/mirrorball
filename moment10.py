import matplotlib.pyplot as plt
import casa_cube as casa
import cmasher as cmr
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import scipy.optimize as optimize

nv = 25
Vsyst = 5.8  # HD163296
win = 7
PA = 154.8 -90. #43.3 # from MAPS (Huang+2018a)
plt.rcParams['font.size'] = '12'

#CO = casa.Cube('~/Observations/exoALMA/CQTau/CQ_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.image.fits')
#CO = casa.Cube('~/Observations/exoALMA/DMTau/DM_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.JvMcorr.fits')
CO = casa.Cube('~/Observations/HD163296-MAPS/HD_163296_CO_220GHz.robust_0.5_wcont.image.fits')

line_profile = np.nansum(CO.image, axis=(1,2)) / CO._beam_area_pix()

iv0 = np.abs(CO.velocity - Vsyst).argmin()
#iv0 = 63 #np.abs(CO.velocity - Vsyst).argmin()
print("systemic channel is ",iv0," v = ",CO.velocity[iv0], ' max v is ',np.max(CO.velocity))

def flip_it(im,PA):
    #rotate to major axis, flip and rotate back
    im = np.fliplr(ndimage.rotate(im,PA+90.,reshape=False))
    return ndimage.rotate(im,-PA-90.-180.,reshape=False)

def get_channel(v):
    iv = np.abs(CO.velocity - v).argmin()
    c1 = np.nan_to_num(CO.image[iv,:,:])
    return c1

def mirror_line_profile_and_get_error(vsys):
    # shift everything so that systemic velocity corresponds to v=0
    vgrid = CO.velocity - vsys

    # flip velocity grid to increasing order if necessary
    if (vgrid[len(vgrid)-1] < vgrid[0]):
       #print("velocity grid is inverted -> flipping")
       vgrid = np.flip(vgrid)
       line_profile_local = np.flip(line_profile)
    else:
       line_profile_local = line_profile

    # plot original line profile
    fig,ax = plt.subplots()
    ax.set_xlim(np.min(vgrid),np.max(vgrid))
    ax.plot(vgrid,line_profile_local)

    # get profile mirrored across the systemic velocity
    neg_channels = vgrid < 0.
    pos_channels = np.logical_not(neg_channels)
    pr = line_profile_local[pos_channels]
    vr = vgrid[pos_channels]

    # reflect the negative channels across the v=0 axis
    pl = np.flip(line_profile_local[neg_channels])
    vl = np.flip(np.abs(vgrid[neg_channels]))

    # get smoothed versions of both profiles
    sl = interpolate.CubicSpline(vl,pl,extrapolate=False)
    sr = interpolate.CubicSpline(vr,pr,extrapolate=False)
    xs = np.linspace(0.,np.max(vgrid),1000)
    pls = sl(xs)
    prs = sr(xs)

    # plot smoothed and reflected profiles and the difference
    ax.plot(xs,pls)
    ax.plot(xs,prs)
    ax.plot(xs,prs-pls)

    # compute the error between the two
    err2 = np.nansum((prs-pls)**2)/np.size(xs)
    filename='lineprofile-vsys-%f.png' % (vsys)
    print("vys = ",vsys,"err2 = ",err2," -> ",filename)
    #plt.show()
    plt.savefig(filename)
    plt.close(fig)
    return err2

def get_vsyst_from_line_profile():
    x0 = np.array([Vsyst])
    res = optimize.minimize(mirror_line_profile_and_get_error,x0,bounds=[[Vsyst-3.,Vsyst+3.]],method='Nelder-Mead')
    print('finished, got vsys = ',res.x,' with error ',res.fun)
    return res.x

def mirror_systemic_channel_and_get_error(PA,iv,plot=False):
    channel = np.nan_to_num(CO.image[iv,:,:])
    mirror  = flip_it(channel,PA[0])
    err = np.mean((channel-mirror)**2)
    print(" channel ",iv," trying PA=",PA,", error is ",err)

    if (plot):
       # make the plot
       fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
       axes[0].imshow(channel,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
       axes[1].imshow(mirror,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
       axes[2].imshow(channel-mirror,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
       plt.show()

    return err

def mirror_channel_and_get_error(PA,iv,iv0,plot=False):
    i = iv - iv0 + int(nv/2)
    iv_sym = iv0 + (nv-1) - int(nv/2) - i
    print('channel ',iv,' goes with ',iv_sym)
    cminus = np.nan_to_num(CO.image[iv,:,:])
    cplus = np.nan_to_num(CO.image[iv_sym,:,:])

    cplusr = flip_it(cplus,PA[0])
    cminusr = flip_it(cminus,PA[0])

    if (plot):
       fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
       axes[0].imshow(cminus,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
       axes[1].imshow(cplusr,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
       axes[2].imshow(cplusr-cminus,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
       plt.savefig('channel%i.png' % i)
       plt.pause(0.5)

    err = np.sqrt(np.mean((cplusr-cminus)**2))
    #print(" channel ",iv," trying PA=",PA,", rms residual is ",err)
    return err

def get_PA(PA0,iv0):
    x0 = np.array(0.)
    offset = 12
    #
    # brute force search for PA to nearest 6 degrees
    #
    PA = optimize.brute(mirror_systemic_channel_and_get_error,args=([iv0]),Ns=30,ranges=[[0.,180.]],finish=None)
    print("brute force search gives PA = ",PA)
    #
    # refine this with a simplex minimization
    #
    x0 = PA
    res = optimize.minimize(mirror_systemic_channel_and_get_error,x0,args=(iv0),method='nelder-mead')
    err = mirror_systemic_channel_and_get_error(res.x,iv0,plot=True)
    print('finished, got PA = ',PA,' with error ',err)
    #
    # try flipped 90 degrees
    #
    PA = res.x[0]
    x0 = PA-90.
    res2 = optimize.minimize(mirror_systemic_channel_and_get_error,x0,args=(iv0),bounds=[[-90.,90.]],method='nelder-mead')
    print('finished, got PA = ',res2.x,' with error ',res2.fun)
    #
    # solve the 90 degree uncertainty with a channel offset from the systemic channel
    #
    err1 = mirror_channel_and_get_error(res.x,iv0+offset,iv0,plot=True)
    err2 = mirror_channel_and_get_error(res2.x,iv0+offset,iv0,plot=True)
    print('comparison, got ',err1,err2,' for PA=',res.x[0],res2.x[0])
    if (err2 < err1):
       PA = res2.x[0]

    return PA

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

Vsyst = get_vsyst_from_line_profile()
iv0 = np.abs(CO.velocity - Vsyst).argmin()
print("closest channel to systemic is ",iv0," v = ",CO.velocity[iv0],CO.velocity[iv0+1])
PA = get_PA(PA,iv0)
print("PA is ",PA," degrees")

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
