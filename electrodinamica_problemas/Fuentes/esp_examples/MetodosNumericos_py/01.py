from vpython import *
#GlowScript 2.7 VPython
grafica = gdisplay(xtitle='Tiempo (s)', ytitle='Angulo (rad)')
posicion = gcurve(color=color.red)
h = 0.01
g = 9.81
l = 0.5
x = pi/5
v = 0.0
def f(x,v):
    return(-g/l*sin(x))

for t in range(1000):
    v=v+f(x,v)*h
    x=x+v*h;
    posicion.plot(pos=(t*h, x))
sleep(0)