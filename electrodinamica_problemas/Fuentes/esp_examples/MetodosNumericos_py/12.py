from vpython import *
#GlowScript 2.7 VPython

from random import random

class particula:
    def __init__(self,indice,x1=0,y1=0,carga=100,col=color.red):
        self.indice = indice
        self.x1=x1; self.y1=y1
        self.carga=carga
        self.color=col
    vx = 0.0; vy = 0.0; masa = 1.0
    x0 = 0.0; x2 = 0.0; y0 = 0.0; y2 = 0.0
    
    def axy(self):
        ax = 0.0; ay = 0.0
        for p in sistema:
            if (p.indice!=self.indice):
                r2=((self.x1-p.x1)*(self.x1-p.x1)
                    +(self.y1-p.y1)*(self.y1-p.y1))
                ax=ax+(self.x1-p.x1)*p.carga*self.carga/pow(r2,1.5)
                ay=ay+(self.y1-p.y1)*p.carga*self.carga/pow(r2,1.5)
        return [ax,ay]

h=0.01; alfa=0.3
f=200; mov=10

fijas=[particula(i+mov,x1=100*cos(2*pi*i/f),y1=100*sin(2*pi*i/f),
       col=color.yellow) for i in range(f)]

moviles=[particula(i,x1=90*random()*cos(2*pi*random()),
        y1=90*random()*sin(2*pi*random()),carga=50)
        for i in range(mov)]
    
sistema=moviles+fijas

for p in fijas:
    sphere(pos=vec(p.x1,p.y1,0),radius=0.8, color=p.color)

esferas=[sphere(pos=vec(p.x1,p.y1,0),radius=1.2, color=p.color) for p in moviles]

for p in moviles:
    p.x0=p.x1-p.vx*h+p.axy()[0]*h*h/2
    p.y0=p.y1-p.vy*h+p.axy()[1]*h*h/2
    
for n in range(3500):
    rate(100)
    for p in moviles:
        fa = alfa*h/2/p.masa-1
        fa2 = 2*p.masa/(alfa*h+2*p.masa);
        p.x2=fa2*(2*p.x1+p.x0*fa+p.axy()[0]*h*h)
        p.y2=fa2*(2*p.y1+p.y0*fa+p.axy()[1]*h*h)
        esferas[p.indice].pos=vec(p.x2,p.y2,0)
        p.x0=p.x1; p.x1=p.x2
        p.y0=p.y1; p.y1=p.y2
    sleep(0)
        