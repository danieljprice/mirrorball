import matplotlib.pyplot as plt
import casa_cube as casa
import cmasher as cmr
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import scipy.optimize as optimize

win = 7
plt.rcParams['font.size'] = '12'

#CO = casa.Cube('~/Observations/exoALMA/CQTau/CQ_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.image.fits')
#CO = casa.Cube('~/Observations/exoALMA/DMTau/DM_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.JvMcorr.fits')
CO = casa.Cube('~/Observations/HD163296-MAPS/HD_163296_CO_220GHz.robust_0.5_wcont.image.fits')

nv = len(CO.velocity)
line_profile = np.nansum(CO.image, axis=(1,2)) / CO._beam_area_pix()

def flip_it(im,PA):
    #rotate to major axis, flip and rotate back
    im = np.fliplr(ndimage.rotate(im,PA+90.,reshape=False))
    return ndimage.rotate(im,-PA-90.-180.,reshape=False)

def get_channel(v):
    iv = np.abs(CO.velocity - v).argmin()

    # do not interpolate past bounds of the array, ends just return end channel
    ivp1 = iv+1
    ivm1 = iv-1
    if (ivp1 > len(CO.velocity) or ivm1 < 0):
       return CO.image[iv,:,:],iv,iv

    # deal with channels in either increasing or decreasing order
    if ((CO.velocity[iv] < v and CO.velocity[ivp1] >= v) or (CO.velocity[ivp1] <= v and CO.velocity[iv] > v)):
       iv1 = ivp1
    else:
       iv1 = ivm1

    c1 = np.nan_to_num(CO.image[iv,:,:])
    c2 = np.nan_to_num(CO.image[iv1,:,:])
    dv = v - CO.velocity[iv]
    deltav = CO.velocity[iv1] - CO.velocity[iv]
    x = dv/deltav
    #print("retrieving channel at v=",v," between ",CO.velocity[iv]," and ",CO.velocity[iv1]," pos = ",x)
    return c1*(1.-x) + x*c2,iv,iv1

def mirror_line_profile_and_get_error(vsys,plot=False):
    # shift everything so that systemic velocity corresponds to v=0
    vgrid = CO.velocity - vsys

    # flip velocity grid to increasing order if necessary
    if (vgrid[len(vgrid)-1] < vgrid[0]):
       #print("velocity grid is inverted -> flipping")
       vgrid = np.flip(vgrid)
       line_profile_local = np.flip(line_profile)
    else:
       line_profile_local = line_profile

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

    # compute the error between the two
    err2 = np.nansum((prs-pls)**2)/np.size(xs)
    filename='lineprofile-vsys-%f.png' % (vsys)
    #print("vys = ",vsys,"err2 = ",err2," -> ",filename)

    # plot smoothed and reflected profiles and the difference
    if (plot):
       fig,ax = plt.subplots()
       ax.set_xlim(np.min(vgrid),np.max(vgrid))
       ax.plot(vgrid,line_profile_local)
       ax.plot(xs,pls)
       ax.plot(xs,prs)
       ax.plot(xs,prs-pls)
       plt.show()
       #plt.savefig(filename)
       plt.close(fig)

    return err2

def get_vsyst_from_line_profile():
    x0 = np.array([Vsyst])
    res = optimize.minimize(mirror_line_profile_and_get_error,x0,bounds=[[Vsyst-3.,Vsyst+3.]],method='Nelder-Mead')
    #mirror_line_profile_and_get_error(res.x,plot=True)
    print('finished, got vsys = ',res.x,' with error ',res.fun)
    return res.x

def mirror_systemic_channel_and_get_error(PA,iv,plot=False,image=None):
    # optional argument image can be used to send an interpolated channel here
    if (image is not None):
       channel = image
    else:
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

def mirror_at_vsys(vsys,PA,plot=False):
    image,iv,iv1 = get_channel(vsys)
    return mirror_systemic_channel_and_get_error(PA,iv,image=image,plot=plot)

def mirror_at_vsys_PA(PA,vsys,plot=False):
    image,iv,iv1 = get_channel(vsys)
    return mirror_systemic_channel_and_get_error(PA,iv,image=image,plot=plot)

def mirror_channel_and_get_error(PA,iv,iv0,plot=False):
    i = iv - iv0 + int(nv/2)
    iv_sym = iv0 + (nv-1) - int(nv/2) - i
    cminus = np.nan_to_num(CO.image[iv,:,:])
    cplus = np.nan_to_num(CO.image[iv_sym,:,:])

    cplusr = flip_it(cplus,PA[0])
    cminusr = flip_it(cminus,PA[0])

    err = np.sqrt(np.mean((cplusr-cminus)**2))
    print('channel ',iv,' goes with ',iv_sym,' trying PA ',PA,' rms is ',err)
    #print(" channel ",iv," trying PA=",PA,", rms residual is ",err)

    if (plot):
       fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
       axes[0].imshow(cminus,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
       axes[1].imshow(cplusr,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
       axes[2].imshow(cplusr-cminus,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
       plt.savefig('channel%i.png' % i)
       plt.pause(0.5)

    return err

def get_PA(iv0,vsys,PA0=None):
    x0 = np.array(0.)
    offset = 12
    if (PA0 is not None):
       PA = PA0
    else:
       #
       # if no initial guess, brute force search for PA to nearest 6 degrees
       #
       PA = optimize.brute(mirror_systemic_channel_and_get_error,args=([iv0]),Ns=30,ranges=[[0.,180.]],finish=None)
       print("brute force search gives PA = ",PA)
       #
       # try flipped 90 degrees
       #
       PA2 = PA-90.
       #res2 = optimize.minimize(mirror_systemic_channel_and_get_error,x0,args=(iv0),bounds=[[-90.,90.]],method='nelder-mead')
       #print('finished, got PA = ',res2.x,' with error ',res2.fun)
       #
       # solve the 90 degree uncertainty with a channel offset from the systemic channel
       #
       err1 = mirror_channel_and_get_error([PA],iv0+offset,iv0,plot=False)
       err2 = mirror_channel_and_get_error([PA2],iv0+offset,iv0,plot=False)
       print('checking 90 degree flip, got ',err1,err2,' for PA=',PA,PA2)
       if (err2 < err1):
          PA = PA2
    #
    # refine the final result with a simplex minimization
    #
    x0 = [PA]
    res = optimize.minimize(mirror_at_vsys_PA,x0,args=(vsys),method='nelder-mead')
    #res = optimize.minimize(mirror_systemic_channel_and_get_error,args=([iv0]),method='nelder-mead')
    err = mirror_at_vsys_PA(res.x,vsys,plot=True)
    #err = mirror_systemic_channel_and_get_error(res.x,iv0,plot=True)
    print('finished, got PA = ',res.x,' with error ',err)

    return res.x[0]

def get_vsyst_from_channels(Vsys0,PA):
    res = optimize.minimize(mirror_at_vsys,Vsys0,args=(PA),bounds=[[Vsys0-0.2,Vsys0+0.2]],method='nelder-mead')
    print('finished, got vsys = ',res.x,' with error ',res.fun)
    mirror_at_vsys(res.x,PA,plot=True)
    return res.x

def plot_interpolated_channel(vsys):
    im,iv,iv1 = get_channel(vsys)
    fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
    axes[0].set_title('V = {}'.format(CO.velocity[iv]))
    axes[0].imshow(CO.image[iv,:,:],vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
    axes[1].set_title('V = {}'.format(vsys))
    axes[1].imshow(im,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
    axes[2].set_title('V = {}'.format(CO.velocity[iv1]))
    axes[2].imshow(CO.image[iv1,:,:],vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
    plt.show()
    return

def mirror_cube(Vsys,PA,plot=False):
    iv0 = np.abs(CO.velocity - Vsys).argmin()
    cube = np.zeros(np.shape(CO.image[:,:,:]))
    #map = np.zeros(np.shape(CO.image[iv0,:,:]))
    for iv in range(2*int(nv/2)-1):
       dv = CO.velocity[iv] - Vsys
       vsym = Vsys - dv
       print('channel ',iv,' V=',CO.velocity[iv],' goes with V=',vsym)
       cminus = np.nan_to_num(CO.image[iv,:,:])
       cplus,iv1,iv2 = get_channel(vsym)
       cplusr = flip_it(cplus,PA)
       if (plot):
          fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
          axes[0].set_title('V={:.2f} km/s, Vsys={:.2f} km/s'.format(CO.velocity[iv]-Vsys,Vsys))
          axes[0].imshow(cminus,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
          axes[1].set_title('V={:.2f} km/s flipped, PA={:.2f} deg'.format(vsym-Vsys,PA))
          axes[1].imshow(cplusr,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
          axes[2].set_title('residual')
          axes[2].imshow(cplusr-cminus,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
          plt.savefig('channel'+str(iv).zfill(3)+'.png')
          #plt.pause(0.5)
          plt.close(fig)

       cube[iv,:,:] = cplusr - cminus

    if (plot):
       import os
       os.system('ffmpeg -i channel%03d.png -r 10 -vb 50M -bt 100M -pix_fmt yuv420p -vf setpts=4.\*PTS mirror-channels.mp4')

    return cube

def get_vsys_and_PA():
    Vsyst = get_vsyst_from_line_profile()
    iv0 = np.abs(CO.velocity - Vsyst).argmin()
    print("closest channel to systemic is ",iv0," v = ",CO.velocity[iv0],CO.velocity[iv0+1])
    #plot_interpolated_channel(Vsyst)

    #PA=43.3
    #Vsyst = get_vsyst_from_channels(Vsyst,PA)

    PA = get_PA(iv0,Vsyst,PA0=45.)
    print("PA is ",PA," degrees")

    Vsyst = get_vsyst_from_channels(Vsyst,[PA])
    PA = get_PA(iv0,Vsyst,PA0=PA)
    print("refined PA is ",PA," degrees")
    return Vsyst,PA

(Vsyst,PA) =(5.763665,42.343013)
#Vsyst,PA = get_vsys_and_PA()
print(" USING Vsyst = %f, PA=%f " %(Vsyst,PA))

CO.image = mirror_cube(Vsyst,PA,plot=True)
print("writing to mirror.fits...")
CO.writeto('mirror.fits')

CO.plot(moment=0,fmin=-0.002,fmax=0.002,limit=0.3,cmap='inferno')
plt.savefig('HD163-moment10.png',bbox_inches='tight')
plt.show()

CO.plot(moment=1,limit=0.3,cmap='RdBu_r')
plt.savefig('HD163-moment101.png',bbox_inches='tight')
plt.show()
