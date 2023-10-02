# mirrorball
Compute the residual from mirroring molecular line cube observations of protoplanetary discs across the systemic channel. Also fits for systemic velocity and position angle.

# requirements
casa_cube (https://github.com/cpinte/casa_cube)

# example script
```
python example.py
```

# basic usage

## fit for systemic velocity and position angle
```
 import mirrorball as mb
 mb = mb.cube('myfile.fits')

 (Vsys,PA) = mb.get_vsys_and_PA()
 print(Vsys,PA)
```

## generate mirror cube and write to fits file
```
 mc = mb.mirror_cube(vsys=Vsys,PA=PA,plot=True)

 print("writing to mirror.fits...")
 mc.writeto('mirror.fits')
```

## plot a mirror moment
```
 mc.plot(moment=8)
```