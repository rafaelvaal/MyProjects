from vpython import *
#GlowScript 2.7 VPython
class particula:
    def __init__(self,indice,x=0,y=0,carga=100,col=color.red):
        self.indice = indice;
        self.x=x; self.y=y
        self.carga=carga
        self.color=col;

fijas=[particula(0,x = -10,y = 0,carga = 100,col=color.red),
       particula(1,x =  10,y = 0,carga = -100,col=color.green)]

esferas=[sphere(pos=vec(p.x,p.y,0),radius=0.5, color=p.color) for p in fijas]

def E(x,y):
    Ex=0; Ey=0
    for p in fijas:
        r2 = (x-p.x)**2+(y-p.y)**2
        Ex = Ex + p.carga*(x-p.x)/r2**1.5
        Ey = Ey + p.carga*(y-p.y)/r2**1.5
    if Ex<0.00001: return 1/0.00001
    else: return (Ey/Ex)

hx = [0.005,-0.005]

for h in hx:
    for i in range(9):
        x1 = 0.0
        y1 = 2.5*i-10.0
        dy2dx=(E(x1+h/2,y1)-E(x1-h/2,y1))/(2*h)   
        y0 = y1-E(x1,y1)*h+h*h*dy2dx/2.0
        para=False;Linea=[vec(x1,y1,0)]
        while (not para):
            dy2dx=(E(x1+h/2,y1)-E(x1-h/2,y1))/(2*h)
            y2 = 2*y1-y0+dy2dx*h*h
            Linea.append(vec(x1,y1,0))
            y0=y1;y1=y2
            x1 = x1 + h
            for m in fijas:
                if ((x1-m.x)**2+(y1-m.y)**2<1): para = True 
        curve(pos=Linea,radius=0.1,color=color.yellow)