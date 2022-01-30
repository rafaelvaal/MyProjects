from vpython import *
#GlowScript 2.7 VPython
grafica = gdisplay(xtitle='Tiempo (s)', ytitle='Angulo (rad)')
posicion = gcurve(color=color.red)
h = 0.01
g = 9.81; l=0.5
x = 0.99*pi; v=0.0
def f(x,v):
    return(-g/l*sin(x))
for t in range(1000):
    k1=h*v
    l1=h*f(x,v)
    k2=h*(v+l1/2)
    l2=h*f(x+k1/2,v+l1/2)
    k3=h*(v-l1+2*l2)
    l3=h*f(x-k1+2*k2,v-l1+2*l2)
    v=v+(l1+4*l2+l3)/6;
    x=x+(k1+4*k2+k3)/6;
    posicion.plot(pos=(t*h, x))