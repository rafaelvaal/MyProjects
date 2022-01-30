from vpython import *
#GlowScript 2.7 VPython
grafica = gdisplay(xtitle='Tiempo (s)', ytitle='Angulo (rad)')
posicion = gcurve(color=color.red)
h=0.01; g=9.81; l=0.5
x1=0.99*pi; v=0.0

def f(x):
    return(-g/l*sin(x))

x0=x1-v*h+h*h*f(x1)/2

for t in range(1000):
    x2=2*x1-x0+h*h*f(x1)
    x0=x1; x1=x2
    posicion.plot(pos=(t*h,x2))