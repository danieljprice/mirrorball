import sys
import os
import mirrorball as mb

if (len(sys.argv) == 2):
   file=sys.argv[1]
else:
   #file='~/Observations/HD142527/HD142527_12CO_contsub.fits'
   #file='~/Observations/exoALMA/HD34282/HD_34282_12CO_robust0.5_width0.028kms_threshold4.0sigma_taper0.15arcsec.clean.image.fits'
   file='~/Observations/exoALMA/final/MWC_758/MWC_758_12CO_robust0.5_width0.100kms_threshold2.0sigma_taper0.1arcsec.clean.image_denoise.fits'
   #file='~/Observations/exoALMA/final/LkCa_15/LkCa_15_12CO_robust1.0_width0.100kms_threshold2.0sigma_taper0.15arcsec.clean.image_denoise.fits'
   #file='~/Observations/exoALMA/final/AA_Tau/AA_Tau_12CO_robust0.5_width0.100kms_threshold2.0sigma_taper0.1arcsec.clean.image_denoise.fits'
   #file='~/Observations/exoALMA/final/J1604-2130/RXJ1604.3-2130_12CO_robust0.0_width0.100kms_threshold3.0sigma_taper0.1arcsec.clean.image.fits'

if (os.path.isfile('mirror.fits')):
   print("Error: mirror.fits already exists, please move or rename")
   exit()

mb = mb.cube(file)

#Vsys = mb.get_vsys_from_line_profile()
#vsys_values,v_values = mb.get_vsys_as_function_of_v(plot=True,filename='J1604-vs-v.pdf')

#(Vsys,PA) = mb.get_vsys_and_PA()
#(Vsys,PA) = (6.293192,152.332042)  # Lk Ca 15
(Vsys,PA) = (5.931398,-29.419161) #  MWC758

PA_values,v_values = mb.get_PA_for_each_channel(vsys=Vsys,PA=PA,plot=True,filename='mwc-PA-vs-v.pdf')

mb.plot_PA_values(PA_values,v_values)

#(Vsys,PA) = (5.931398,-29.419161) #  MWC758
#(Vsys,PA) = (-2.339460,27.737111) # HD34282
#(Vsys,PA) = (4.126982,74.243033) # SY Cha denoise
#(Vsys,PA) = (4.147810,74.062892) # SY Cha
#(Vsys,PA) = (4.502893,54.819273) # IM Lup
#(Vsys,PA) = (4.4,53.) # IM Lup literature
#(Vsys,PA) = (5.715284,66.469651) # PDS 70
#(Vsys,PA) = (6.220097,145.735504) # CQ Tau
#(Vsys,PA) =(5.763665,42.343013) # HD 163296
#(Vsys,PA) = (6.045766,65.414138) # DM Tau

#(Vsys,PA) = (6.293192,152.332042)  # Lk Ca 15
#print(" USING (Vsys,PA) = (%f,%f) " %(Vsys,PA))

#mc = mb.mirror_cube(vsys=6.293192,PA=152.332042,plot=True)

#CO.image = mb.mirror_cube(Vsys,PA,plot=True)
#print("writing to mirror.fits...")

#mc.writeto('mirror.fits')

#CO.plot(moment=8,limit=5.,Tb=True,fmin=2.)
#plt.savefig('mirror-moment8.png',bbox_inches='tight')
#plt.show()

#CO.plot(moment=9,limit=5.,fmin=Vsys-2.,fmax=Vsys+2.,cmap='RdBu_r')
#plt.savefig('mirror-moment9.png',bbox_inches='tight')
#plt.show()
