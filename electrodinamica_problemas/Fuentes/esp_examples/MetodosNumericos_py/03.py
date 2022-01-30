from vpython import *
#GlowScript 2.7 VPython
grafica = gdisplay(xtitle='Tiempo (s)', ytitle='Angulo (radianes)', background=color.white,foreground=color.black)
posicion = gcurve(color=color.red)
h = 0.01
g = 9.81; l = 0.5
x = pi/5; v=0.0
def f(x,v):
    return (-g/l*sin(x)-0.6*v)
for t in range(1000):
    k1=h*v;
    l1=h*f(x,v)
    k2=h*(v+l1)
    l2=h*f(x+k1,v+l1)
    v=v+(l1+l2)/2.0;
    x=x+(k1+k2)/2.0;
    posicion.plot(pos=(t*h, x))