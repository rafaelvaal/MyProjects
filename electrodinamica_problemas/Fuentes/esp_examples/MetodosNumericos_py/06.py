from vpython import *
#GlowScript 2.7 VPython
grafica = gdisplay(xtitle='Angulo (rad)', ytitle='Velocidad (rad/s)', background=color.white,foreground=color.black)
posicion = gcurve(color=color.red)
h=0.01
g=9.81; l=0.5
x=0.99*pi; v=0.0
def f(x):
    return(-g/l*sin(x))
for t in range(1000):
    x0=x;
    x=x+h*v+h*h*f(x)/2
    v=v+h*(f(x0)+f(x))/2
    posicion.plot(pos=(x,v))