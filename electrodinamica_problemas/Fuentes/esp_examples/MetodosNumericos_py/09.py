from vpython import *
#GlowScript 2.7 VPython
class particula:
    def __init__(self,indice): self.indice = indice
    vx = 0; vy = 0
    m = 1
    x0=0.0; x1=0.0; x2=0.0
    y0=0.0; y1=0.0; y2=0.0
    colores=color.red
    def axy(self):
        ax=0.0; ay=0.0;
        for p in sistema:
            if (p.indice!=self.indice):
                r2=((self.x1-p.x1)*(self.x1-p.x1)
                    +(self.y1-p.y1)*(self.y1-p.y1))
                ax=ax-(self.x1-p.x1)*p.m/pow(r2,1.5)
                ay=ay-(self.y1-p.y1)*p.m/pow(r2,1.5)
        return [ax,ay]

display(range=1.75)

sistema=[particula(i) for i in range(3)]
h=0.001

sistema[0].colores=color.red
sistema[1].colores=color.green
sistema[2].colores=color.yellow

sistema[0].x1=0.97000436; sistema[0].y1=-0.24308753
sistema[1].x1=-0.97000436; sistema[1].y1=0.24308753
sistema[2].x1=0.0; sistema[2].y1=0.0

sistema[0].vx=0.466203685; sistema[0].vy=0.43236573
sistema[1].vx=0.466203685; sistema[1].vy=0.43236573
sistema[2].vx=-0.93240737; sistema[2].vy=-0.86473146

esferas=[sphere(pos=vec(p.x1,p.y1,0.0),radius=0.08,make_trail=True,
                color=p.colores) for p in sistema]

flechas=[arrow(pos=vec(p.x1,p.y1,0.0),shaftwidth=0.03,
                color=p.colores,axis=vec(0.5*p.vx,0.5*p.vy,0.0)) for p in sistema]

for p in sistema:
    p.x0=p.x1-p.vx*h+p.axy()[0]*h*h/2
    p.y0=p.y1-p.vy*h+p.axy()[1]*h*h/2

while True:
    rate(150)
    for p in sistema:
        p.x2=2*p.x1-p.x0+p.axy()[0]*h*h
        p.y2=2*p.y1-p.y0+p.axy()[1]*h*h
        p.vx=(p.x2-p.x1)/h; p.vy=(p.y2-p.y1)/h
        esferas[p.indice].pos=vec(p.x2,p.y2,0.0)
        flechas[p.indice].pos=vec(p.x2,p.y2,0.0)
        flechas[p.indice].axis=vec(0.5*p.vx,0.5*p.vy,0.0)
        p.x0=p.x1; p.x1=p.x2
        p.y0=p.y1; p.y1=p.y2
    sleep(0)