from vpython import *
#GlowScript 2.7 VPython
ventana1 = gdisplay(xtitle='Tiempo', ytitle='Posicion')
ventana2 = gdisplay(xtitle='Tiempo', ytitle='Energias')
posicion = gcurve(color=color.red, gdisplay=ventana1)
energiaP = gcurve(color=color.blue,gdisplay=ventana2)
energiaC = gcurve(color=color.red, gdisplay=ventana2)
energiaT = gcurve(color=color.green,gdisplay=ventana2)
k = 3.0; m = 1.0; t = 0.0
x1 = 5.0; v = 0.0; h = 0.01
E0 = (k*x1*x1+m*v*v)/2
x0 = x1-v*h+h*h*(-k*x1)

for i in range(1000):
    posicion.plot( pos=(t,v) )
    t = t + h
    x2 = 2*x1-x0+(-k*x1)*h*h
    vd = 2/m*(E0-k*x2*x2/2)
    if vd>=0: v = sign(x2-x1)*sqrt(vd)
    else: v=0
    energiaP.plot(pos=(t, (k*x2*x2)/2))
    energiaC.plot(pos=(t, (m*v*v)/2))
    energiaT.plot(pos=(t, (k*x2*x2+m*v*v)/2))
    x0=x1; x1=x2