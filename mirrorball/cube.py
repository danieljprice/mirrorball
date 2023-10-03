import matplotlib.pyplot as plt
import casa_cube as casa
import cmasher as cmr
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import sys
import os

class cube:

    def __init__(self, file=None, **kwargs):
        """
           initialise a mirrorball cube. Arguments:

           file : filename
        """
        # Correct path if needed
        file = os.path.normpath(os.path.expanduser(file))
        self.file = file

        # Read line cube using casa_cube
        try:
            print("opening "+self.file)
            self.CO = casa.Cube(self.file,**kwargs)
            self.nv = len(self.CO.velocity)
            print("got %i channels" %(self.nv-1))

            if (self.nv > 1):
               self.dvchan = 1.0*(self.CO.velocity[1] - self.CO.velocity[0])
               print(' channel spacing is ',self.dvchan,' km/s')

            self.line_profile = np.nansum(self.CO.image, axis=(1,2)) / self.CO._beam_area_pix()

        except OSError:
            print('cannot open', self.file)


    def flip_it(self,im,PA):
        """
           rotate a velocity channel to match the channel on the other side of the disc
        """
        #rotate to major axis, flip and rotate back
        im = np.fliplr(ndimage.rotate(im,PA+90.,reshape=False))
        return ndimage.rotate(im,-PA-90.-180.,reshape=False)

    def get_channel(self,v):
        """
           get channel corresponding to specified velocity, by interpolating from neighbouring channels
        """
        iv = np.abs(self.CO.velocity - v).argmin()
    
        # do not interpolate past bounds of the array, ends just return end channel
        ivp1 = iv+1
        ivm1 = iv-1
        if (ivp1 > len(self.CO.velocity)-1 or ivm1 < 0):
           return self.CO.image[iv,:,:],iv,iv
    
        # deal with channels in either increasing or decreasing order
        if ((self.CO.velocity[iv] < v and self.CO.velocity[ivp1] >= v) or (self.CO.velocity[ivp1] <= v and self.CO.velocity[iv] > v)):
           iv1 = ivp1
        else:
           iv1 = ivm1
    
        c1 = np.nan_to_num(self.CO.image[iv,:,:])
        c2 = np.nan_to_num(self.CO.image[iv1,:,:])
        dv = v - self.CO.velocity[iv]
        deltav = self.CO.velocity[iv1] - self.CO.velocity[iv]
        x = dv/deltav
        #print("retrieving channel at v=",v," between ",CO.velocity[iv]," and ",CO.velocity[iv1]," pos = ",x)
        return c1*(1.-x) + x*c2,iv,iv1
    
    def mirror_line_profile_and_get_error(self,vsys,vmin=None,vmax=None,plot=False):
        """
           reflect the line profile across an axis corresponding to the systemic velocity
           and subtract one side from the other to get the residual. The idea is that
           the systemic velocity is the one which minimises this residual
        """
        # shift everything so that systemic velocity corresponds to v=0
        vgrid = self.CO.velocity - vsys
    
        # flip velocity grid to increasing order if necessary
        if (vgrid[len(vgrid)-1] < vgrid[0]):
           #print("velocity grid is inverted -> flipping")
           vgrid = np.flip(vgrid)
           line_profile_local = np.flip(self.line_profile)
        else:
           line_profile_local = self.line_profile
    
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

        if (vmin is not None and vmax is not None):
           xcut = xs[(xs > vmin) & (xs < vmax)]
        else:
           xcut = xs
        pls = sl(xcut)
        prs = sr(xcut)
    
        # compute the error between the two
        err2 = np.nansum((prs-pls)**2)/np.size(xs)
        filename='lineprofile-vsys-%f.png' % (vsys)
        #print("vys = ",vsys,"err2 = ",err2," -> ",filename)
    
        # plot smoothed and reflected profiles and the difference
        if (plot):
           fig,ax = plt.subplots()
           ax.set_xlim(np.min(vgrid),np.max(vgrid))
           ax.plot(vgrid,line_profile_local)
           ax.plot(xcut,pls,c='green')
           ax.plot(-xcut,pls,c='green')
           ax.plot(xcut,prs,c='orange')
           ax.plot(xcut,prs-pls,c='red')
           ax.axvline(0., color='grey', linestyle='dotted')
           plt.show()
           #plt.savefig(filename)
           plt.close(fig)
    
        return err2
    
    def get_vsys_from_line_profile(self,dvtry=None,vmin=None,vmax=None,plot=True):
        """
           fit for the systemic velocity (defined here as the symmetry axis for the line profile)
        """
        # initial guess
        vsys = np.mean(self.CO.velocity)  # a wild guess
        # print(" initial guess is ",vsys,self.CO.velocity)
 
        # use flat prior +/- 3 km/s from mean velocity by default
        if (dvtry is None):
           dvtry = 3.

        bounds = [[vsys-dvtry,vsys+dvtry]]
        x0 = np.array([vsys])
        res = optimize.minimize(self.mirror_line_profile_and_get_error,x0,args=(vmin,vmax),bounds=bounds,method='Nelder-Mead')
        self.mirror_line_profile_and_get_error(res.x,vmin=vmin,vmax=vmax,plot=plot)
        print(' got vsys = ',res.x,' with error ',res.fun, ' using v from ',vmin,' to ',vmax)
        return res.x

    def get_vsys_as_function_of_v(self,plot=False,filename='vsys-vs-v.pdf'):
        vsys_values = []
        dvtry = 3.
        deltav = self.dvchan
        v_values = np.arange(deltav,dvtry,deltav)
        for v in v_values:
            vsysi = self.get_vsys_from_line_profile(dvtry=dvtry,vmin=v,vmax=v+deltav,plot=True)
            vsys_values.append(vsysi)

        vsys_tot = self.get_vsys_from_line_profile(dvtry=dvtry,plot=False)

        if (plot):
           fig,ax = plt.subplots()
           ax.plot(v_values,vsys_values)
           #ax.errorbar(v_values,vsys_values,yerr=verr_values)
           ax.set_xlabel('v-vsys [km/s]')
           ax.set_ylabel('vsys [km/s]')
           ax.axhline(vsys_tot, color='red', linestyle='dotted')
           print("saving to ",filename)
           plt.savefig(filename,bbox_inches='tight')
           plt.show()
           plt.close(fig)
        
        return vsys_values,v_values
    
    def mirror_systemic_channel_and_get_error(self,PA,iv,plot=False,image=None):
        """
           mirror the systemic velocity channel, which in
           a symmetric disc should be the reflection of itself
        """
        # optional argument image can be used to send an interpolated channel here
        if (image is not None):
           channel = image
        else:
           channel = np.nan_to_num(self.CO.image[iv,:,:])
    
        mirror  = self.flip_it(channel,PA[0])
        err = np.mean((channel-mirror)**2)
        print(" channel ",iv," trying PA=",PA,", error is ",err)
    
        if (plot):
           # make the plot

           win = 7
           plt.rcParams['font.size'] = '12'
           fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
           axes[0].imshow(channel,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
           axes[1].imshow(mirror,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
           axes[2].imshow(channel-mirror,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
           plt.show()
    
        return err
    
    def mirror_at_vsys(self,vsys,PA,plot=False):
        """
           mirror the image at the systemic velocity (using an interpolated channel)
        """
        image,iv,iv1 = self.get_channel(vsys)
        return self.mirror_systemic_channel_and_get_error(PA,iv,image=image,plot=plot)
    
    def mirror_at_vsys_PA(self,PA,vsys,plot=False):
        """
           as above but takes arguments in different order for use in optimisation
        """
        image,iv,iv1 = self.get_channel(vsys)
        return self.mirror_systemic_channel_and_get_error(PA,iv,image=image,plot=plot)
    
    def mirror_channel_and_get_error(self,PA,iv,iv0,plot=False):
        """
           compare and subtract a channel from its mirror pair across the symmetry axis
        """
        i = iv - iv0 + int(self.nv/2)
        iv_sym = iv0 + (self.nv-1) - int(self.nv/2) - i
        cminus = np.nan_to_num(self.CO.image[iv,:,:])
        cplus = np.nan_to_num(self.CO.image[iv_sym,:,:])
    
        cplusr = self.flip_it(cplus,PA[0])
        cminusr = self.flip_it(cminus,PA[0])
    
        err = np.sqrt(np.mean((cplusr-cminus)**2))
        print('channel ',iv,' goes with ',iv_sym,' trying PA ',PA,' rms is ',err)
        #print(" channel ",iv," trying PA=",PA,", rms residual is ",err)
    
        if (plot):
           win = 7
           plt.rcParams['font.size'] = '12'
           fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
           axes[0].imshow(cminus,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
           axes[1].imshow(cplusr,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
           axes[2].imshow(cplusr-cminus,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower')
           plt.savefig('channel%i.png' % i)
           plt.pause(0.5)
    
        return err
    
    def get_PA(self,iv0,vsys,PA0=None):
        """
           fitting procedure to obtain the position angle of the symmetry axis
        """
        x0 = np.array(0.)
        offset = 12
        if (PA0 is not None):
           PA = PA0
        else:
           #
           # if no initial guess, brute force search for PA to nearest 6 degrees
           #
           PA = optimize.brute(self.mirror_systemic_channel_and_get_error,args=([iv0]),Ns=30,ranges=[[0.,180.]],finish=None)
           print("brute force search gives PA = ",PA)
           #
           # try flipped 90 degrees
           #
           PA2 = PA-90.
           res2 = optimize.minimize(self.mirror_systemic_channel_and_get_error,[PA2],args=(iv0),bounds=[[-90.,90.]],method='nelder-mead')
           print('finished, got PA = ',res2.x,' with error ',res2.fun)
           PA2 = res2.x[0]
           #
           # solve the 90 degree uncertainty with a channel offset from the systemic channel
           #
           err1 = self.mirror_channel_and_get_error([PA],iv0+offset,iv0,plot=False)
           err2 = self.mirror_channel_and_get_error([PA2],iv0+offset,iv0,plot=False)
           print('checking 90 degree flip, got ',err1,err2,' for PA=',PA,PA2)
           if (err2 < err1):
              PA = PA2
        #
        # refine the final result with a simplex minimization
        #
        x0 = [PA]
        res = optimize.minimize(self.mirror_at_vsys_PA,x0,args=(vsys),method='nelder-mead')
        #res = optimize.minimize(mirror_systemic_channel_and_get_error,args=([iv0]),method='nelder-mead')
        err = self.mirror_at_vsys_PA(res.x,vsys,plot=True)
        #err = mirror_systemic_channel_and_get_error(res.x,iv0,plot=True)
        print('finished, got PA = ',res.x,' with error ',err)
    
        return res.x[0]
    
    def get_vsys_from_channels(self,vsys0,PA,vsys0_err=0.2):
        """
           fit for the systemic velocity by reflecting the image of the systemic channel about the symmetry axis
           this should give more precise results than just reflecting the line profile
        """
        res = optimize.minimize(self.mirror_at_vsys,vsys0,args=(PA),bounds=[[vsys0-vsys0_err,vsys0+vsys0_err]],method='nelder-mead')
        print('finished, got vsys = ',res.x,' with error ',res.fun)
        self.mirror_at_vsys(res.x,PA,plot=True)
        return res.x
    
    def plot_interpolated_channel(self,vsys):
        """
           sanity check: plot an interpolated channel compared to the two neighbouring channels
        """
        im,iv,iv1 = self.get_channel(vsys)
        win = 7
        fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
        axes[0].set_title('V = {}'.format(self.CO.velocity[iv]))
        axes[0].imshow(self.CO.image[iv,:,:],vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
        axes[1].set_title('V = {}'.format(vsys))
        axes[1].imshow(im,vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
        axes[2].set_title('V = {}'.format(self.CO.velocity[iv1]))
        axes[2].imshow(self.CO.image[iv1,:,:],vmin=0.,vmax=0.04,cmap='viridis',origin='lower')
        plt.show()
        return
    
    def mirror_cube(self,vsys=None,PA=None,plot=False):
        """
           compute the residual of a line cube by subtracting the mirror opposite of each channel
        """
        if (vsys is None and self.vsys is not None):
           vsys = self.vsys
        if (PA is None and self.PA is not None):
           PA = self.PA
        
        if (vsys is None or PA is None):
           self.get_vsys_and_PA()
           
        iv0 = np.abs(self.CO.velocity - vsys).argmin()
        cube = np.zeros(np.shape(self.CO.image[:,:,:]))
        #map = np.zeros(np.shape(CO.image[iv0,:,:]))
        for iv in range(2*int(self.nv/2)-1):
           dv = self.CO.velocity[iv] - vsys
           vsym = vsys - dv
           print('channel ',iv,' V=',self.CO.velocity[iv],' goes with V=',vsym)
           channel = np.nan_to_num(self.CO.image[iv,:,:])
           channel_opposite,iv1,iv2 = self.get_channel(vsym)
           channel_opposite_flipped = self.flip_it(channel_opposite,PA)
           if (plot):
              win = 7
              fig, axes = plt.subplots(1,3, figsize=(21,8), sharex='all', sharey='all', num=win)
              pix_scale = self.CO.pixelscale
              xlabel = r'$\Delta$ RA (")'
              ylabel = r'$\Delta$ Dec (")'
              xaxis_factor = -1
              #limit = 5. # arcsec
              halfsize = np.asarray(channel.shape) / 2 * pix_scale
              extent = [-halfsize[0]*xaxis_factor,halfsize[0]*xaxis_factor, -halfsize[1], halfsize[1]]
              axes[0].set_xlabel(xlabel)
              axes[0].set_ylabel(ylabel)
              #axes[0].set_xlim(limit,-limit)
              #axes[0].set_ylim(-limit,limit)
              axes[0].set_title('V={:.2f} km/s, vsys={:.2f} km/s'.format(self.CO.velocity[iv]-vsys,vsys))
              axes[0].imshow(channel,vmin=0.,vmax=0.04,cmap='viridis',origin='lower',extent=extent)
              axes[1].set_xlabel(xlabel)
              #axes[0].set_xlim(limit,-limit)
              #axes[0].set_ylim(-limit,limit)
              axes[1].set_title('V={:.2f} km/s flipped, PA={:.2f} deg'.format(vsym-vsys,PA))
              axes[1].imshow(channel_opposite_flipped,vmin=0.,vmax=0.04,cmap='viridis',origin='lower',extent=extent)
              axes[2].set_xlabel(xlabel)
              #axes[0].set_xlim(limit,-limit)
              #axes[0].set_ylim(-limit,limit)
              axes[2].set_title('residual')
              axes[2].imshow(channel-channel_opposite_flipped,vmin=-0.01,vmax=0.01,cmap='inferno_r',origin='lower',extent=extent)
              plt.savefig('channel'+str(iv).zfill(3)+'.png')
              #plt.pause(0.5)
              plt.close(fig)
    
           cube[iv,:,:] = channel - channel_opposite_flipped
    
        if (plot):
           import os
           os.system('ffmpeg -i channel%03d.png -r 10 -vb 50M -bt 100M -pix_fmt yuv420p -vf setpts=4.\*PTS mirror-channels.mp4')
    
        mc = self.CO
        mc.image = cube
        return mc
    
    def get_vsys_and_PA(self):
        """
           fit both the systemic velocity and position angle
           with increasing degrees of sophistication
        """
        self.vsys = self.get_vsys_from_line_profile()
        iv0 = np.abs(self.CO.velocity - self.vsys).argmin()
        print("closest channel to systemic is ",iv0," v = ",self.CO.velocity[iv0],self.CO.velocity[iv0+1])
        #plot_interpolated_channel(vsyst)
    
        #self.vsys = self.get_vsys_from_channels(vsyst,PA)
    
        self.PA = self.get_PA(iv0,self.vsys)
        print("PA is ",self.PA," degrees")
    
        vsyst = self.get_vsys_from_channels(self.vsys,[self.PA])
        self.PA = self.get_PA(iv0,vsys=vsyst,PA0=self.PA)
        self.vsys = vsyst[0]
        print("refined PA is ",self.PA," degrees")
        return vsyst[0],self.PA
