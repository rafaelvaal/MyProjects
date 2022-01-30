from vpython import *
#GlowScript 2.7 VPython

from random import random

class particula:
    def __init__(self,indice,x=0,y=0,carga=100,col=color.red):
        self.indice = indice
        self.x=x; self.y=y
        self.carga=carga
        self.color=col
    
h = 1.0; Ep2=0.0; f=200; mov=10

fijas=[particula(i+mov,x=100*cos(2*pi*i/f),y=100*sin(2*pi*i/f),
       col=color.yellow) for i in range(f)]

moviles=[particula(i,x=90.0*random()*cos(2*pi*random()),
        y=90.0*random()*sin(2*pi*random()),carga=50)
        for i in range(mov)]
    
sistema=moviles+fijas;

for p in fijas:
    sphere(pos=vec(p.x,p.y,0),radius=0.8, color=p.color)

esferas=[sphere(pos=vec(p.x,p.y,0),radius=1.2, color=p.color) for p in moviles]

def Ep():
    e=0;
    for k1 in sistema:
        for k2 in sistema:
            if (k1.indice!=k2.indice):
                e=e+1/((k1.x-k2.x)*(k1.x-k2.x)+(k1.y-k2.y)*(k1.y-k2.y))
    return(e)

while (Ep()-Ep2!=0):
    rate(50)
    Ep2=Ep()
    for p in moviles:
        D = {Ep():[p.x,p.y]}
        x1=p.x; y1= p.y
        for k in range(8):
            xp=x1+h*cos(2*pi*k/8)
            yp=y1+h*sin(2*pi*k/8)
            p.x=xp; p.y=yp
            D[Ep()]=[xp,yp]
        p.x=D[min(D)][0]       
        p.y=D[min(D)][1]
        esferas[p.indice].pos=vec(p.x,p.y,0)
print ('Energia Potencial final = ', Ep())